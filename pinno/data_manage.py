"""
Módulo para gestión y muestreo de datos en PINNs.

Este módulo se encarga de la generación de datos sintéticos (puntos de colocación,
condiciones iniciales y de frontera) y de la carga de datos experimentales externos.
Actúa como una fábrica de datos que alimenta el proceso de entrenamiento.
"""

# pinno/data_manage.py
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

class DataManager:
    """Gestor de datos con soporte para CSV y Mapeo"""
    
    def __init__(self, config: Dict[str, Any], problem_name: str, 
                 csv_data: Optional[pd.DataFrame] = None, 
                 column_mapping: Optional[Dict[str, str]] = None):
        self.config = config
        self.problem_name = problem_name
        self.physics_config = config["PHYSICS_CONFIG"]
        self.data_config = config["DATA_CONFIG"]
        self.csv_data = csv_data
        self.column_mapping = column_mapping
        self.use_csv = csv_data is not None and column_mapping is not None
        
        self.sampling_strategies = {
            "SHO": self._sample_sho_data,
            "DHO": self._sample_sho_data,
            "HEAT": self._sample_heat_data
        }
        
        self.csv_sampling_strategies = {
            "SHO": self._sample_sho_csv_data,
            "DHO": self._sample_sho_csv_data,
            "HEAT": self._sample_heat_csv_data
        }

    def prepare_data(self):
        if self.use_csv:
            sampling_fn = self.csv_sampling_strategies.get(self.problem_name)
        else:
            sampling_fn = self.sampling_strategies.get(self.problem_name)
            
        if not sampling_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        self.training_data = sampling_fn()

    def get_training_data(self) -> Dict[str, tf.Tensor]:
        return self.training_data

    # --- Estrategias Analiticas ---
    def _sample_sho_data(self) -> Dict[str, tf.Tensor]:
        t_domain = self.physics_config["t_domain"]
        n_coll = self.data_config["n_collocation"]
        t_coll = self._sample_uniform(t_domain, n_coll, 1)
        t0 = self._sample_initial_time(t_domain[0])
        x0_true = tf.constant(self.physics_config["initial_conditions"]["x0"], dtype=tf.float32)
        v0_true = tf.constant(self.physics_config["initial_conditions"]["v0"], dtype=tf.float32)
        return {"t_coll": t_coll, "t0": t0, "x0_true": x0_true, "v0_true": v0_true}

    def _sample_heat_data(self) -> Dict[str, tf.Tensor]:
        x_d, y_d, t_d = self.physics_config["x_domain"], self.physics_config["y_domain"], self.physics_config["t_domain"]
        n_c, n_i, n_b = self.data_config["n_collocation"], self.data_config["n_initial"], self.data_config["n_boundary"]
        
        return {
            "xyt_coll": self._sample_3d_uniform(x_d, y_d, t_d, n_c),
            "xyt0": self._sample_initial_condition_heat(x_d, y_d, n_i),
            "xyt_b": self._sample_boundary_condition_heat(x_d, y_d, t_d, n_b)
        }

    # --- Estrategias CSV ---
    def _sample_sho_csv_data(self) -> Dict[str, tf.Tensor]:
        # Usamos el mapeo para sacar las columnas correctas
        t_col = self.column_mapping['time']
        x_col = self.column_mapping['displacement']
        
        t_data = self.csv_data[t_col].values.reshape(-1, 1)
        x_data = self.csv_data[x_col].values.reshape(-1, 1)
        
        # Puntos de colocacion en el rango del CSV
        t_min, t_max = np.min(t_data), np.max(t_data)
        t_coll = self._sample_uniform([t_min, t_max], self.data_config["n_collocation"], 1)
        
        return {
            "t_coll": t_coll,
            "t_data": tf.constant(t_data, dtype=tf.float32),
            "x_data": tf.constant(x_data, dtype=tf.float32)
        }

    def _sample_heat_csv_data(self) -> Dict[str, tf.Tensor]:
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

    # --- Auxiliares ---
    def _sample_uniform(self, domain, n, dim=1):
        return tf.convert_to_tensor(np.random.uniform(domain[0], domain[1], (n, dim)), dtype=tf.float32)

    def _sample_3d_uniform(self, dx, dy, dt, n):
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        return tf.convert_to_tensor(np.hstack([x, y, t]), dtype=tf.float32)

    def _sample_initial_time(self, t0):
        return tf.convert_to_tensor(np.array([[t0]]), dtype=tf.float32)

    def _sample_initial_condition_heat(self, dx, dy, n):
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.zeros_like(x)
        return tf.convert_to_tensor(np.hstack([x, y, t]), dtype=tf.float32)

    def _sample_boundary_condition_heat(self, dx, dy, dt, n):
        # Simplificado: bordes aleatorios
        x = np.random.uniform(dx[0], dx[1], (n, 1))
        y = np.random.uniform(dy[0], dy[1], (n, 1))
        t = np.random.uniform(dt[0], dt[1], (n, 1))
        mask = np.random.randint(0, 4, n)
        x[mask==0] = dx[0]; x[mask==1] = dx[1]
        y[mask==2] = dy[0]; y[mask==3] = dy[1]
        return tf.convert_to_tensor(np.hstack([x, y, t]), dtype=tf.float32)