"""
Módulo para gestión y muestreo de datos en PINNs.

Este módulo se encarga de la generación de datos sintéticos (puntos de colocación,
condiciones iniciales y de frontera) y de la carga de datos experimentales externos.
Actúa como una fábrica de datos que alimenta el proceso de entrenamiento.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


class DataManager:
    """
    Gestor centralizado de datos para el entrenamiento de PINNs.

    Esta clase implementa el patrón Factory para seleccionar la estrategia de muestreo
    adecuada según el problema físico (SHO, HEAT, etc.) y gestiona la inyección
    de datos externos si se proporcionan.

    Attributes:
        config (Dict[str, Any]): Configuración completa del experimento.
        problem_name (str): Identificador del problema ("SHO", "HEAT", etc.).
        physics_config (dict): Configuración específica de la física (dominios, constantes).
        data_config (dict): Configuración de cantidad de puntos a muestrear.
        external_data (Optional[pd.DataFrame]): DataFrame con datos cargados externamente.
        training_data (Dict[str, tf.Tensor]): Diccionario final con los tensores listos para entrenar.
    """
    
    def __init__(self, config: Dict[str, Any], problem_name: str):
        """
        Inicializa el DataManager.

        Args:
            config (Dict[str, Any]): Diccionario de configuración global.
            problem_name (str): Nombre del problema a resolver.
        """
        self.config = config
        self.problem_name = problem_name
        self.physics_config = config["PHYSICS_CONFIG"]
        self.data_config = config["DATA_CONFIG"]
        
        # Mapeo de estrategias de muestreo
        self.sampling_strategies = {
            "SHO": self._sample_sho_data,
            "DHO": self._sample_sho_data,
            "HEAT": self._sample_heat_data
        }
        
        self.external_data: Optional[pd.DataFrame] = None
        self.training_data: Dict[str, tf.Tensor] = {}

    def load_external_data(self, filepath: str) -> bool:
        """
        Carga datos experimentales desde un archivo CSV o XLSX.

        Este método lee el archivo, lo almacena en `self.external_data` y prepara
        el gestor para inyectar estos datos en el conjunto de entrenamiento.

        Args:
            filepath (str): Ruta relativa o absoluta al archivo de datos.

        Returns:
            bool: True si la carga fue exitosa, False si ocurrió un error.

        Raises:
            ValueError: Si el formato del archivo no es .csv o .xlsx.
        """
        try:
            if filepath.endswith('.csv'):
                self.external_data = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx'):
                self.external_data = pd.read_excel(filepath)
            else:
                raise ValueError("Formato no soportado. Use .csv o .xlsx")
            return True
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False

    def prepare_data(self):
        """
        Orquesta la preparación de todos los datos necesarios.

        1. Ejecuta la estrategia de muestreo sintético (colocación/frontera).
        2. Si existen datos externos cargados, los procesa e inyecta.

        Raises:
            ValueError: Si el `problem_name` no tiene una estrategia asociada.
        """
        sampling_fn = self.sampling_strategies.get(self.problem_name)
        if not sampling_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
        self.training_data = sampling_fn()
        
        # Inyectar datos externos si existen
        if self.external_data is not None:
            self._inject_external_data()

    def _inject_external_data(self):
        """
        Convierte y mapea el DataFrame externo a tensores de TensorFlow.

        Identifica las columnas del archivo (ej. 't', 'x' para SHO o 'x','y','t','u' para HEAT)
        y agrega los tensores resultantes al diccionario `self.training_data` bajo
        las claves "ext_...".
        """
        df = self.external_data
        try:
            if self.problem_name in ["SHO", "DHO"]:
                if 't' in df.columns and 'x' in df.columns:
                    t_ext = tf.convert_to_tensor(df['t'].values.reshape(-1, 1), dtype=tf.float32)
                    x_ext = tf.convert_to_tensor(df['x'].values.reshape(-1, 1), dtype=tf.float32)
                    self.training_data["ext_t"] = t_ext
                    self.training_data["ext_x"] = x_ext
                    print("Datos externos (t, x) inyectados exitosamente.")
            
            elif self.problem_name == "HEAT":
                if all(col in df.columns for col in ['x', 'y', 't', 'u']):
                    xyt_ext = df[['x', 'y', 't']].values.astype(np.float32)
                    u_ext = df[['u']].values.astype(np.float32)
                    self.training_data["ext_xyt"] = tf.convert_to_tensor(xyt_ext)
                    self.training_data["ext_u"] = tf.convert_to_tensor(u_ext)
                    print("Datos externos (x,y,t -> u) inyectados exitosamente.")
        except Exception as e:
            print(f"Advertencia: No se pudieron inyectar datos externos: {e}")

    def get_training_data(self) -> Dict[str, tf.Tensor]:
        """
        Devuelve los datos listos para el entrenamiento.

        Returns:
            Dict[str, tf.Tensor]: Diccionario conteniendo tensores de colocación,
            condiciones iniciales, de frontera y datos externos (si aplican).
        """
        return self.training_data

    # --- Estrategias de Muestreo ---

    def _sample_sho_data(self) -> Dict[str, tf.Tensor]:
        """Genera datos para Osciladores Armónicos (1D tiempo)."""
        t_domain = self.physics_config["t_domain"]
        n_coll = self.data_config["n_collocation"]
        
        t_coll = self._sample_uniform(t_domain, n_coll, 1)
        t0 = self._sample_initial_time(t_domain[0])
        
        x0_true = tf.constant(self.physics_config["initial_conditions"]["x0"], dtype=tf.float32)
        v0_true = tf.constant(self.physics_config["initial_conditions"]["v0"], dtype=tf.float32)
        
        return {
            "t_coll": t_coll, 
            "t0": t0, 
            "x0_true": x0_true, 
            "v0_true": v0_true
        }

    def _sample_heat_data(self) -> Dict[str, tf.Tensor]:
        """Genera datos para la Ecuación de Calor 2D (x, y, t)."""
        x_domain = self.physics_config["x_domain"]
        y_domain = self.physics_config["y_domain"]
        t_domain = self.physics_config["t_domain"]
        
        n_coll = self.data_config["n_collocation"]
        n_init = self.data_config["n_initial"]
        n_bound = self.data_config["n_boundary"]
        
        xyt_coll = self._sample_3d_uniform(x_domain, y_domain, t_domain, n_coll)
        xyt0 = self._sample_initial_condition_heat(x_domain, y_domain, n_init)
        xyt_b = self._sample_boundary_condition_heat(x_domain, y_domain, t_domain, n_bound)
        
        return {
            "xyt_coll": xyt_coll, 
            "xyt0": xyt0, 
            "xyt_b": xyt_b
        }

    # --- Métodos Auxiliares de Muestreo ---

    def _sample_uniform(self, domain, n_points, dim=1):
        """Muestreo uniforme en un dominio simple."""
        points = np.random.uniform(domain[0], domain[1], size=(n_points, dim)).astype(np.float32)
        return tf.convert_to_tensor(points, dtype=tf.float32)

    def _sample_3d_uniform(self, dx, dy, dt, n):
        """Muestreo uniforme en un dominio 3D (x, y, t)."""
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        return tf.convert_to_tensor(np.hstack([x, y, t]).astype(np.float32))
        
    def _sample_initial_time(self, t0):
        """Crea un tensor para el tiempo inicial t0."""
        return tf.convert_to_tensor(np.array([[t0]], dtype=np.float32))

    def _sample_initial_condition_heat(self, dx, dy, n):
        """Muestrea puntos espaciales aleatorios en t=0."""
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.zeros_like(x)
        return tf.convert_to_tensor(np.hstack([x, y, t]).astype(np.float32))

    def _sample_boundary_condition_heat(self, dx, dy, dt, n):
        """
        Muestrea puntos en los bordes del dominio espacial rectangular.
        
        Distribuye aleatoriamente puntos en x=0, x=L, y=0, y=L a lo largo del tiempo.
        """
        # Implementación simplificada para muestreo aleatorio de bordes
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        
        # Forzar aleatoriamente a bordes
        mask = np.random.randint(0, 4, n)
        # Borde x=min
        x[mask==0] = dx[0]
        # Borde x=max
        x[mask==1] = dx[1]
        # Borde y=min
        y[mask==2] = dy[0]
        # Borde y=max
        y[mask==3] = dy[1]
        
        return tf.convert_to_tensor(np.hstack([x, y, t]).astype(np.float32))