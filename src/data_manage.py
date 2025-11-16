"""
Módulo para gestión y muestreo de datos en PINNs.
(Versión 0.0.4 - Con soporte para mapeo de columnas CSV)
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


class DataManager:
    """
    Gestor de datos para PINNs con soporte para mapeo de columnas CSV.
    """
    
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
        
        # Estrategias de muestreo
        self.sampling_strategies = {
            "SHO": self._sample_sho_data,
            "DHO": self._sample_sho_data,
            "HEAT": self._sample_heat_data
        }
        
        # Estrategias de muestreo para datos CSV
        self.csv_sampling_strategies = {
            "SHO": self._sample_sho_csv_data,
            "DHO": self._sample_sho_csv_data,
            "HEAT": self._sample_heat_csv_data
        }

    def prepare_data(self):
        """Prepara todos los datos necesarios para el entrenamiento"""
        if self.use_csv:
            sampling_fn = self.csv_sampling_strategies.get(self.problem_name)
        else:
            sampling_fn = self.sampling_strategies.get(self.problem_name)
            
        if not sampling_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        self.training_data = sampling_fn()

    def get_training_data(self) -> Dict[str, tf.Tensor]:
        """Devuelve los datos de entrenamiento en formato estandarizado"""
        return self.training_data

    def _sample_sho_data(self) -> Dict[str, tf.Tensor]:
        """Muestrea datos para problemas SHO/DHO analíticos"""
        t_domain = self.physics_config["t_domain"]
        n_coll = self.data_config["n_collocation"]
        
        # Puntos de colocación
        t_coll = self._sample_uniform(t_domain, n_coll, 1)
        
        # Condiciones iniciales
        t0 = self._sample_initial_time(t_domain[0])
        x0_true = tf.constant(self.physics_config["initial_conditions"]["x0"], 
                             dtype=tf.float32)
        v0_true = tf.constant(self.physics_config["initial_conditions"]["v0"], 
                             dtype=tf.float32)
        
        return {
            "t_coll": t_coll,
            "t0": t0,
            "x0_true": x0_true,
            "v0_true": v0_true
        }

    def _sample_heat_data(self) -> Dict[str, tf.Tensor]:
        """Muestrea datos para la ecuación de calor 2D analítica"""
        x_domain = self.physics_config["x_domain"]
        y_domain = self.physics_config["y_domain"]
        t_domain = self.physics_config["t_domain"]
        
        n_coll = self.data_config["n_collocation"]
        n_init = self.data_config["n_initial"]
        n_bound = self.data_config["n_boundary"]
        
        # Puntos de colocación (x, y, t) internos
        xyt_coll = self._sample_3d_uniform(x_domain, y_domain, t_domain, n_coll)
        
        # Condición inicial (t=0)
        xyt0 = self._sample_initial_condition_heat(x_domain, y_domain, n_init)
        
        # Condiciones de contorno (en los bordes espaciales)
        xyt_b = self._sample_boundary_condition_heat(x_domain, y_domain, t_domain, n_bound)
        
        return {
            "xyt_coll": xyt_coll,
            "xyt0": xyt0,
            "xyt_b": xyt_b
        }

    def _sample_sho_csv_data(self) -> Dict[str, tf.Tensor]:
        """Muestrea datos para problemas SHO/DHO desde CSV usando mapeo de columnas"""
        # Extraer datos del CSV usando el mapeo
        time_col = self.column_mapping['time']
        disp_col = self.column_mapping['displacement']
        
        t_data = self.csv_data[time_col].values.reshape(-1, 1)
        x_data = self.csv_data[disp_col].values.reshape(-1, 1)
        
        # Generar puntos de colocación dentro del rango temporal del CSV
        t_min, t_max = np.min(t_data), np.max(t_data)
        n_coll = self.data_config["n_collocation"]
        t_coll = self._sample_uniform([t_min, t_max], n_coll, 1)
        
        return {
            "t_coll": tf.constant(t_coll, dtype=tf.float32),
            "t_data": tf.constant(t_data, dtype=tf.float32),
            "x_data": tf.constant(x_data, dtype=tf.float32)
        }

    def _sample_heat_csv_data(self) -> Dict[str, tf.Tensor]:
        """Muestrea datos para Heat Equation desde CSV usando mapeo de columnas"""
        # Extraer datos del CSV usando el mapeo
        x_col = self.column_mapping['x']
        y_col = self.column_mapping['y']
        time_col = self.column_mapping['time']
        temp_col = self.column_mapping['temperature']
        
        x_data = self.csv_data[x_col].values.reshape(-1, 1)
        y_data = self.csv_data[y_col].values.reshape(-1, 1)
        t_data = self.csv_data[time_col].values.reshape(-1, 1)
        u_data = self.csv_data[temp_col].values.reshape(-1, 1)
        
        # Generar puntos de colocación dentro del dominio del CSV
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        t_min, t_max = np.min(t_data), np.max(t_data)
        
        n_coll = self.data_config["n_collocation"]
        xyt_coll = self._sample_3d_uniform([x_min, x_max], [y_min, y_max], [t_min, t_max], n_coll)
        
        return {
            "xyt_coll": tf.constant(xyt_coll, dtype=tf.float32),
            "x_data": tf.constant(x_data, dtype=tf.float32),
            "y_data": tf.constant(y_data, dtype=tf.float32),
            "t_data": tf.constant(t_data, dtype=tf.float32),
            "u_data": tf.constant(u_data, dtype=tf.float32)
        }

    def _sample_uniform(self, domain: Tuple[float, float], 
                       n_points: int, dim: int = 1) -> np.ndarray:
        """Muestrea uniformemente en un dominio 1D"""
        low, high = domain
        points = np.random.uniform(low, high, size=(n_points, dim)).astype(np.float32)
        return points

    def _sample_3d_uniform(self, domain_x: Tuple[float, float], 
                          domain_y: Tuple[float, float],
                          domain_t: Tuple[float, float], 
                          n_points: int) -> np.ndarray:
        """Muestrea uniformemente en un dominio 3D (x, y, t)"""
        x_points = np.random.uniform(domain_x[0], domain_x[1], size=(n_points, 1))
        y_points = np.random.uniform(domain_y[0], domain_y[1], size=(n_points, 1))
        t_points = np.random.uniform(domain_t[0], domain_t[1], size=(n_points, 1))
        xyt_points = np.hstack([x_points, y_points, t_points]).astype(np.float32)
        return xyt_points

    def _sample_initial_time(self, t0: float) -> tf.Tensor:
        """Muestrea el tiempo inicial"""
        t0_arr = np.array([[t0]], dtype=np.float32)
        return tf.constant(t0_arr, dtype=tf.float32)

    def _sample_initial_condition_heat(self, x_domain: Tuple[float, float], 
                                      y_domain: Tuple[float, float], 
                                      n_points: int) -> tf.Tensor:
        """Muestrea condiciones iniciales para la ecuación de calor (t=0)"""
        x_points = np.random.uniform(x_domain[0], x_domain[1], size=(n_points, 1))
        y_points = np.random.uniform(y_domain[0], y_domain[1], size=(n_points, 1))
        t0_points = np.zeros_like(x_points)
        xyt0_points = np.hstack([x_points, y_points, t0_points]).astype(np.float32)
        return tf.constant(xyt0_points, dtype=tf.float32)

    def _sample_boundary_condition_heat(self, x_domain: Tuple[float, float], 
                                       y_domain: Tuple[float, float],
                                       t_domain: Tuple[float, float], 
                                       n_points: int) -> tf.Tensor:
        """Muestrea condiciones de contorno para la ecuación de calor (bordes espaciales)"""
        n_per_edge = max(1, n_points // 4)
        points_list = []
        
        # Borde x=0
        x0 = np.zeros((n_per_edge, 1))
        y0 = np.random.uniform(y_domain[0], y_domain[1], size=(n_per_edge, 1))
        t0 = np.random.uniform(t_domain[0], t_domain[1], size=(n_per_edge, 1))
        points_list.append(np.hstack([x0, y0, t0]))
        
        # Borde x=L
        xL = np.full((n_per_edge, 1), x_domain[1])
        yL = np.random.uniform(y_domain[0], y_domain[1], size=(n_per_edge, 1))
        tL = np.random.uniform(t_domain[0], t_domain[1], size=(n_per_edge, 1))
        points_list.append(np.hstack([xL, yL, tL]))
        
        # Borde y=0
        x1 = np.random.uniform(x_domain[0], x_domain[1], size=(n_per_edge, 1))
        y1 = np.zeros((n_per_edge, 1))
        t1 = np.random.uniform(t_domain[0], t_domain[1], size=(n_per_edge, 1))
        points_list.append(np.hstack([x1, y1, t1]))
        
        # Borde y=L
        x2 = np.random.uniform(x_domain[0], x_domain[1], size=(n_per_edge, 1))
        y2 = np.full((n_per_edge, 1), y_domain[1])
        t2 = np.random.uniform(t_domain[0], t_domain[1], size=(n_per_edge, 1))
        points_list.append(np.hstack([x2, y2, t2]))
        
        remaining_points = n_points - (n_per_edge * 4)
        if remaining_points > 0:
            x0_extra = np.zeros((remaining_points, 1))
            y0_extra = np.random.uniform(y_domain[0], y_domain[1], size=(remaining_points, 1))
            t0_extra = np.random.uniform(t_domain[0], t_domain[1], size=(remaining_points, 1))
            points_list.append(np.hstack([x0_extra, y0_extra, t0_extra]))
        
        xyt_boundary = np.vstack(points_list).astype(np.float32)
        return tf.constant(xyt_boundary, dtype=tf.float32)