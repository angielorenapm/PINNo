# src/data_manage.py
import numpy as np
import tensorflow as tf
import pandas as pd  # NUEVO: Para leer csv/excel
from typing import Dict, Any, Tuple, List, Optional

class DataManager:
    
    def __init__(self, config: Dict[str, Any], problem_name: str):
        self.config = config
        self.problem_name = problem_name
        self.physics_config = config["PHYSICS_CONFIG"]
        self.data_config = config["DATA_CONFIG"]
        
        self.sampling_strategies = {
            "SHO": self._sample_sho_data,
            "DHO": self._sample_sho_data,
            "HEAT": self._sample_heat_data
        }
        
        # Almacén para datos cargados externamente
        self.external_data: Optional[pd.DataFrame] = None

    # --- NUEVO: Método para cargar datos externos ---
    def load_external_data(self, filepath: str) -> bool:
        """Carga datos desde CSV o XLSX para entrenamiento."""
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
        sampling_fn = self.sampling_strategies.get(self.problem_name)
        if not sampling_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
        self.training_data = sampling_fn()
        
        # Si existen datos externos, inyectarlos al diccionario de entrenamiento
        if self.external_data is not None:
            self._inject_external_data()

    def _inject_external_data(self):
        """Convierte el DataFrame cargado en tensores y lo agrega a training_data."""
        df = self.external_data
        # Lógica simple de mapeo de columnas basada en nombres esperados
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
        return self.training_data

    # --- Métodos existentes de muestreo (sin cambios mayores) ---
    def _sample_sho_data(self) -> Dict[str, tf.Tensor]:
        t_domain = self.physics_config["t_domain"]
        n_coll = self.data_config["n_collocation"]
        t_coll = self._sample_uniform(t_domain, n_coll, 1)
        t0 = self._sample_initial_time(t_domain[0])
        x0_true = tf.constant(self.physics_config["initial_conditions"]["x0"], dtype=tf.float32)
        v0_true = tf.constant(self.physics_config["initial_conditions"]["v0"], dtype=tf.float32)
        return {"t_coll": t_coll, "t0": t0, "x0_true": x0_true, "v0_true": v0_true}

    def _sample_heat_data(self) -> Dict[str, tf.Tensor]:
        x_domain = self.physics_config["x_domain"]
        y_domain = self.physics_config["y_domain"]
        t_domain = self.physics_config["t_domain"]
        n_coll = self.data_config["n_collocation"]
        n_init = self.data_config["n_initial"]
        n_bound = self.data_config["n_boundary"]
        
        xyt_coll = self._sample_3d_uniform(x_domain, y_domain, t_domain, n_coll)
        xyt0 = self._sample_initial_condition_heat(x_domain, y_domain, n_init)
        xyt_b = self._sample_boundary_condition_heat(x_domain, y_domain, t_domain, n_bound)
        
        return {"xyt_coll": xyt_coll, "xyt0": xyt0, "xyt_b": xyt_b}

    # (Helpers de muestreo random se mantienen igual que en tu original)
    def _sample_uniform(self, domain, n_points, dim=1):
        points = np.random.uniform(domain[0], domain[1], size=(n_points, dim)).astype(np.float32)
        return tf.convert_to_tensor(points, dtype=tf.float32)

    def _sample_3d_uniform(self, dx, dy, dt, n):
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        return tf.convert_to_tensor(np.hstack([x, y, t]).astype(np.float32))
        
    def _sample_initial_time(self, t0):
        return tf.convert_to_tensor(np.array([[t0]], dtype=np.float32))

    def _sample_initial_condition_heat(self, dx, dy, n):
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.zeros_like(x)
        return tf.convert_to_tensor(np.hstack([x, y, t]).astype(np.float32))

    def _sample_boundary_condition_heat(self, dx, dy, dt, n):
        # Implementación simplificada para ahorrar espacio aquí, usar la lógica original completa
        # Genera puntos en los bordes del cubo 3D
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        # Forzar aleatoriamente a bordes (ejemplo simplificado)
        mask = np.random.randint(0, 4, n)
        x[mask==0] = dx[0]; x[mask==1] = dx[1]
        y[mask==2] = dy[0]; y[mask==3] = dy[1]
        return tf.convert_to_tensor(np.hstack([x, y, t]).astype(np.float32))