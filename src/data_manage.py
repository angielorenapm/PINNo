# src/data_manage.py
"""
Módulo para gestión y muestreo de datos en PINNs.

Responsabilidades:
- Muestrear puntos de colocación, iniciales y de contorno
- Gestionar diferentes esquemas de muestreo por problema
- Proporcionar datos en formato consistente para el entrenamiento

Pattern: Factory Method para diferentes estrategias de muestreo
"""
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, List


class DataManager:
    """
    Gestor de datos para PINNs.
    
    Pattern: Factory + Strategy para esquemas de muestreo
    """
    
    def __init__(self, config: Dict[str, Any], problem_name: str):
        self.config = config
        self.problem_name = problem_name
        self.physics_config = config["PHYSICS_CONFIG"]
        self.data_config = config["DATA_CONFIG"]
        
        # Estrategias de muestreo por problema
        self.sampling_strategies = {
            "SHO": self._sample_sho_data,
            "DHO": self._sample_sho_data,  # Misma estrategia que SHO
            "WAVE": self._sample_wave_data
        }

    def prepare_data(self):
        """Prepara todos los datos necesarios para el entrenamiento"""
        sampling_fn = self.sampling_strategies.get(self.problem_name)
        if not sampling_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        self.training_data = sampling_fn()

    def get_training_data(self) -> Dict[str, tf.Tensor]:
        """Devuelve los datos de entrenamiento en formato estandarizado"""
        return self.training_data

    def _sample_sho_data(self) -> Dict[str, tf.Tensor]:
        """Muestrea datos para problemas SHO/DHO"""
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

    def _sample_wave_data(self) -> Dict[str, tf.Tensor]:
        """Muestrea datos para ecuación de onda"""
        x_domain = self.physics_config["x_domain"]
        t_domain = self.physics_config["t_domain"]
        
        n_coll = self.data_config["n_collocation"]
        n_init = self.data_config["n_initial"]
        n_bound = self.data_config["n_boundary"]
        
        # Puntos de colocación (espacio-tiempo)
        xt_coll = self._sample_2d_uniform(x_domain, t_domain, n_coll)
        
        # Condiciones iniciales (t=0)
        xt0 = self._sample_initial_condition_wave(x_domain, n_init)
        
        # Condiciones de contorno (x=0 y x=L)
        xt_b = self._sample_boundary_condition_wave(x_domain, t_domain, n_bound)
        
        return {
            "xt_coll": xt_coll,
            "xt0": xt0,
            "xt_b": xt_b
        }

    def _sample_uniform(self, domain: Tuple[float, float], 
                       n_points: int, dim: int = 1) -> tf.Tensor:
        """Muestrea uniformemente en un dominio 1D"""
        low, high = domain
        points = np.random.uniform(low, high, size=(n_points, dim)).astype(np.float32)
        return tf.convert_to_tensor(points, dtype=tf.float32)

    def _sample_2d_uniform(self, domain_x: Tuple[float, float], 
                          domain_t: Tuple[float, float], 
                          n_points: int) -> tf.Tensor:
        """Muestrea uniformemente en un dominio 2D"""
        x_points = np.random.uniform(domain_x[0], domain_x[1], size=(n_points, 1))
        t_points = np.random.uniform(domain_t[0], domain_t[1], size=(n_points, 1))
        xt_points = np.hstack([x_points, t_points]).astype(np.float32)
        return tf.convert_to_tensor(xt_points, dtype=tf.float32)

    def _sample_initial_time(self, t0: float) -> tf.Tensor:
        """Muestrea el tiempo inicial"""
        t0_arr = np.array([[t0]], dtype=np.float32)
        return tf.convert_to_tensor(t0_arr, dtype=tf.float32)

    def _sample_initial_condition_wave(self, x_domain: Tuple[float, float], 
                                      n_points: int) -> tf.Tensor:
        """Muestrea condiciones iniciales para la ecuación de onda"""
        x_points = np.random.uniform(x_domain[0], x_domain[1], size=(n_points, 1))
        t0_points = np.zeros_like(x_points)
        xt0_points = np.hstack([x_points, t0_points]).astype(np.float32)
        return tf.convert_to_tensor(xt0_points, dtype=tf.float32)

    def _sample_boundary_condition_wave(self, x_domain: Tuple[float, float], 
                                       t_domain: Tuple[float, float], 
                                       n_points: int) -> tf.Tensor:
        """Muestrea condiciones de contorno para la ecuación de onda"""
        n_half = n_points // 2
        
        # Contorno izquierdo (x = x_min)
        t_left = np.random.uniform(t_domain[0], t_domain[1], size=(n_half, 1))
        x_left = np.full_like(t_left, fill_value=x_domain[0])
        xt_left = np.hstack([x_left, t_left])
        
        # Contorno derecho (x = x_max)  
        t_right = np.random.uniform(t_domain[0], t_domain[1], 
                                   size=(n_points - n_half, 1))
        x_right = np.full_like(t_right, fill_value=x_domain[1])
        xt_right = np.hstack([x_right, t_right])
        
        # Combinar
        xt_boundary = np.vstack([xt_left, xt_right]).astype(np.float32)
        return tf.convert_to_tensor(xt_boundary, dtype=tf.float32)