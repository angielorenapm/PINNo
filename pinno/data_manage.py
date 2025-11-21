# pinno/data_manage.py
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
    Gestor centralizado para la generación de datos de entrenamiento (sintéticos y experimentales).
    
    Selecciona y ejecuta estrategias de muestreo dependiendo del problema físico y
    de si se proporcionan datos externos (CSV).
    """
    
    def __init__(self, config: Dict[str, Any], problem_name: str, 
                 csv_data: Optional[pd.DataFrame] = None, 
                 column_mapping: Optional[Dict[str, str]] = None):
        """
        Inicializa el gestor de datos y configura las estrategias de muestreo.

        Args:
            config (Dict[str, Any]): Diccionario de configuración global del experimento.
            problem_name (str): Identificador del problema físico ("SHO", "HEAT", etc.).
            csv_data (Optional[pd.DataFrame]): DataFrame con datos cargados externamente (para modo Data-Driven).
            column_mapping (Optional[Dict[str, str]]): Diccionario que mapea nombres de columnas físicas a las del CSV.
        """
        self.config = config
        self.problem_name = problem_name
        self.physics_config = config["PHYSICS_CONFIG"]
        self.data_config = config["DATA_CONFIG"]
        self.csv_data = csv_data
        self.column_mapping = column_mapping
        self.use_csv = csv_data is not None and column_mapping is not None
        
        # Estrategias para modo Analítico (Sin datos externos)
        self.sampling_strategies = {
            "SHO": self._sample_sho_data,
            "DHO": self._sample_sho_data,
            "HEAT": self._sample_heat_data
        }
        
        # Estrategias para modo Data-Driven (Con CSV)
        self.csv_sampling_strategies = {
            "SHO": self._sample_sho_csv_data,
            "DHO": self._sample_sho_csv_data,
            "HEAT": self._sample_heat_csv_data
        }

    def prepare_data(self):
        """
        Ejecuta la estrategia de muestreo correspondiente para generar los tensores de entrenamiento.

        Selecciona entre estrategia CSV o Analítica basándose en la inicialización.

        Raises:
            ValueError: Si el `problem_name` no tiene una estrategia de muestreo definida.
        """
        if self.use_csv:
            sampling_fn = self.csv_sampling_strategies.get(self.problem_name)
        else:
            sampling_fn = self.sampling_strategies.get(self.problem_name)
            
        if not sampling_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        self.training_data = sampling_fn()

    def get_training_data(self) -> Dict[str, tf.Tensor]:
        """
        Devuelve el conjunto de datos procesado y listo para el entrenamiento.

        Returns:
            Dict[str, tf.Tensor]: Diccionario con los tensores de entrenamiento 
            (ej. ``t_coll``, ``x0_true``, ``t_data``).
        """
        return self.training_data

    # --- Estrategias Analiticas ---

    def _sample_sho_data(self) -> Dict[str, tf.Tensor]:
        """
        Genera datos sintéticos para problemas 1D (SHO/DHO) en modo analítico.

        Returns:
            Dict[str, tf.Tensor]: Diccionario con:
            - ``t_coll``: Puntos de colocación en el dominio.
            - ``t0``: Tiempo inicial (t=0).
            - ``x0_true``, ``v0_true``: Condiciones iniciales exactas.
        """
        t_domain = self.physics_config["t_domain"]
        n_coll = self.data_config["n_collocation"]
        
        t_coll = self._sample_uniform(t_domain, n_coll, 1)
        t0 = self._sample_initial_time(t_domain[0])
        
        # Extraer condiciones iniciales del config
        ic = self.physics_config.get("initial_conditions", {"x0": 1.0, "v0": 0.0})
        x0_true = tf.constant(ic["x0"], dtype=tf.float32)
        v0_true = tf.constant(ic["v0"], dtype=tf.float32)
        
        return {
            "t_coll": t_coll, 
            "t0": t0, 
            "x0_true": x0_true, 
            "v0_true": v0_true
        }

    def _sample_heat_data(self) -> Dict[str, tf.Tensor]:
        """
        Genera datos sintéticos para la Ecuación de Calor 2D en modo analítico.

        Returns:
            Dict[str, tf.Tensor]: Diccionario con:
            - ``xyt_coll``: Puntos de colocación espacio-temporales.
            - ``xyt0``: Puntos en t=0 (Condición inicial).
            - ``xyt_b``: Puntos en los bordes espaciales (Condición de frontera).
        """
        x_d = self.physics_config["x_domain"]
        y_d = self.physics_config["y_domain"]
        t_d = self.physics_config["t_domain"]
        
        n_c = self.data_config["n_collocation"]
        n_i = self.data_config["n_initial"]
        n_b = self.data_config["n_boundary"]
        
        return {
            "xyt_coll": self._sample_3d_uniform(x_d, y_d, t_d, n_c),
            "xyt0": self._sample_initial_condition_heat(x_d, y_d, n_i),
            "xyt_b": self._sample_boundary_condition_heat(x_d, y_d, t_d, n_b)
        }

    # --- Estrategias CSV ---

    def _sample_sho_csv_data(self) -> Dict[str, tf.Tensor]:
        """
        Prepara datos para problemas 1D (SHO/DHO) usando datos externos CSV.

        Returns:
            Dict[str, tf.Tensor]: Diccionario con:
            - ``t_coll``: Puntos de colocación generados dentro del rango temporal del CSV.
            - ``t_data``: Tensor de tiempos reales del CSV.
            - ``x_data``: Tensor de desplazamientos reales del CSV.
        """
        # Usamos el mapeo para sacar las columnas correctas
        t_col = self.column_mapping['time']
        x_col = self.column_mapping['displacement']
        
        t_data = self.csv_data[t_col].values.reshape(-1, 1)
        x_data = self.csv_data[x_col].values.reshape(-1, 1)
        
        # Puntos de colocacion en el rango del CSV para la perdida física
        t_min, t_max = np.min(t_data), np.max(t_data)
        t_coll = self._sample_uniform([t_min, t_max], self.data_config["n_collocation"], 1)
        
        return {
            "t_coll": t_coll,
            "t_data": tf.constant(t_data, dtype=tf.float32),
            "x_data": tf.constant(x_data, dtype=tf.float32)
        }

    def _sample_heat_csv_data(self) -> Dict[str, tf.Tensor]:
        """
        Prepara datos para la Ecuación de Calor 2D usando datos externos CSV.

        Returns:
            Dict[str, tf.Tensor]: Diccionario con:
            - ``xyt_coll``: Puntos de colocación 3D dentro del dominio del CSV.
            - ``x_data``, ``y_data``, ``t_data``: Coordenadas reales.
            - ``u_data``: Temperatura real medida.
        """
        x_col = self.column_mapping['x']
        y_col = self.column_mapping['y']
        t_col = self.column_mapping['time']
        u_col = self.column_mapping['temperature']
        
        x_data = self.csv_data[x_col].values.reshape(-1, 1)
        y_data = self.csv_data[y_col].values.reshape(-1, 1)
        t_data = self.csv_data[t_col].values.reshape(-1, 1)
        u_data = self.csv_data[u_col].values.reshape(-1, 1)
        
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        t_min, t_max = np.min(t_data), np.max(t_data)
        
        xyt_coll = self._sample_3d_uniform([x_min, x_max], [y_min, y_max], [t_min, t_max], self.data_config["n_collocation"])
        
        return {
            "xyt_coll": xyt_coll,
            "x_data": tf.constant(x_data, dtype=tf.float32),
            "y_data": tf.constant(y_data, dtype=tf.float32),
            "t_data": tf.constant(t_data, dtype=tf.float32),
            "u_data": tf.constant(u_data, dtype=tf.float32)
        }

    # --- Métodos Auxiliares de Muestreo ---

    def _sample_uniform(self, domain: Tuple[float, float], n: int, dim: int = 1) -> tf.Tensor:
        """
        Genera puntos aleatorios distribuidos uniformemente en un dominio.

        Args:
            domain (Tuple[float, float]): Límites inferior y superior del dominio ``(min, max)``.
            n (int): Número de puntos a generar.
            dim (int, optional): Dimensión de los puntos generados. Por defecto es 1.

        Returns:
            tf.Tensor: Tensor de forma ``(n, dim)`` con tipo ``float32``.
        """
        points = np.random.uniform(domain[0], domain[1], size=(n, dim)).astype(np.float32)
        return tf.convert_to_tensor(points, dtype=tf.float32)

    def _sample_3d_uniform(self, dx: Tuple[float, float], dy: Tuple[float, float], 
                          dt: Tuple[float, float], n: int) -> tf.Tensor:
        """
        Genera puntos de colocación uniformes en un dominio espacio-temporal 3D (x, y, t).

        Args:
            dx (Tuple[float, float]): Dominio espacial en eje X.
            dy (Tuple[float, float]): Dominio espacial en eje Y.
            dt (Tuple[float, float]): Dominio temporal.
            n (int): Número total de puntos a generar.

        Returns:
            tf.Tensor: Tensor de forma ``(n, 3)`` donde las columnas son ``[x, y, t]``.
        """
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        return tf.convert_to_tensor(np.hstack([x, y, t]), dtype=tf.float32)

    def _sample_initial_time(self, t0: float) -> tf.Tensor:
        """
        Crea un tensor constante para el tiempo inicial.

        Args:
            t0 (float): Valor del tiempo inicial (usualmente 0.0).

        Returns:
            tf.Tensor: Tensor de forma ``(1, 1)`` conteniendo el valor ``t0``.
        """
        return tf.convert_to_tensor(np.array([[t0]]), dtype=tf.float32)

    def _sample_initial_condition_heat(self, dx: Tuple[float, float], 
                                      dy: Tuple[float, float], n: int) -> tf.Tensor:
        """
        Muestrea condiciones iniciales para la Ecuación de Calor (t=0).
        
        Genera puntos espaciales (x, y) aleatorios y fija t=0.

        Args:
            dx (Tuple[float, float]): Dominio espacial X.
            dy (Tuple[float, float]): Dominio espacial Y.
            n (int): Número de puntos iniciales.

        Returns:
            tf.Tensor: Tensor de forma ``(n, 3)`` con columnas ``[x, y, 0.0]``.
        """
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.zeros_like(x)
        return tf.convert_to_tensor(np.hstack([x, y, t]), dtype=tf.float32)

    def _sample_boundary_condition_heat(self, dx: Tuple[float, float], 
                                       dy: Tuple[float, float], 
                                       dt: Tuple[float, float], n: int) -> tf.Tensor:
        """
        Muestrea condiciones de frontera para un dominio rectangular 2D a lo largo del tiempo.
        
        Distribuye aleatoriamente los puntos entre los 4 bordes espaciales:
        * Borde Izquierdo (x=min)
        * Borde Derecho (x=max)
        * Borde Inferior (y=min)
        * Borde Superior (y=max)

        Args:
            dx (Tuple[float, float]): Dominio espacial X.
            dy (Tuple[float, float]): Dominio espacial Y.
            dt (Tuple[float, float]): Dominio temporal.
            n (int): Número total de puntos de frontera.

        Returns:
            tf.Tensor: Tensor de forma ``(n, 3)`` con puntos situados en los bordes.
        """
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        
        # Máscara aleatoria para asignar cada punto a uno de los 4 bordes
        mask = np.random.randint(0, 4, n)
        
        # Aplicar restricciones de borde
        x[mask==0] = dx[0]  # x = x_min
        x[mask==1] = dx[1]  # x = x_max
        y[mask==2] = dy[0]  # y = y_min
        y[mask==3] = dy[1]  # y = y_max
        
        return tf.convert_to_tensor(np.hstack([x, y, t]), dtype=tf.float32)