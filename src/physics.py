"""
Módulo que define los problemas físicos y sus ecuaciones diferenciales.
(Versión 0.0.4 - Con soporte para mapeo de columnas CSV)
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

# --- Clase Base Abstracta ---
class PhysicsProblem(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_config = config['PHYSICS_CONFIG']
        self.has_analytical = True
        self.csv_data = None
        self.column_mapping = None  # Nuevo: mapeo de columnas

    @abstractmethod
    def pde_residual(self, model: tf.keras.Model, points: tf.Tensor):
        pass
    
    @abstractmethod
    def analytical_solution(self, points) -> np.ndarray:
        pass
    
    def set_csv_data(self, csv_data: pd.DataFrame, column_mapping: Dict[str, str]):
        """Set CSV data with column mapping for data-driven training"""
        self.csv_data = csv_data
        self.column_mapping = column_mapping
        self.has_analytical = False
    
    def get_training_data(self) -> Optional[Tuple]:
        """Get CSV data formatted for training using column mapping"""
        if self.csv_data is None or self.column_mapping is None:
            return None
        return self._extract_data_with_mapping()

    def _extract_data_with_mapping(self) -> Optional[Tuple]:
        """Extract data using column mapping - to be implemented by subclasses"""
        return None

# --- Implementaciones de Problemas Físicos ---

class SimpleHarmonicOscillator(PhysicsProblem):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.omega = tf.constant(self.domain_config['omega'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, t: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            x_t = tape.gradient(x, t)
        x_tt = tape.gradient(x_t, t)
        del tape
        
        if x_tt is None:
            raise ValueError("El cálculo de la segunda derivada (x_tt) falló y devolvió None.")
                             
        return x_tt + (self.omega**2) * x
    
    def analytical_solution(self, t) -> np.ndarray:
        t_val = t.numpy() if hasattr(t, 'numpy') else t
        x0 = self.domain_config['initial_conditions']['x0']
        v0 = self.domain_config['initial_conditions']['v0']
        omega_val = self.omega.numpy()
        return x0 * np.cos(omega_val * t_val) + (v0 / omega_val) * np.sin(omega_val * t_val)
    
    def _extract_data_with_mapping(self) -> Optional[Tuple]:
        """Extract data using column mapping for SHO"""
        time_col = self.column_mapping.get('time')
        disp_col = self.column_mapping.get('displacement')
        
        if not time_col or not disp_col:
            raise ValueError("Column mapping missing for SHO. Required: 'time', 'displacement'")
        
        t_data = self.csv_data[time_col].values.reshape(-1, 1)
        x_data = self.csv_data[disp_col].values.reshape(-1, 1)
        
        return t_data, x_data

class DampedHarmonicOscillator(SimpleHarmonicOscillator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.zeta = tf.constant(self.domain_config['zeta'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, t: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            x_t = tape.gradient(x, t)
        x_tt = tape.gradient(x_t, t)
        del tape

        if x_tt is None or x_t is None:
             raise ValueError("El cálculo de derivadas para DHO falló.")

        return x_tt + 2 * self.zeta * self.omega * x_t + (self.omega**2) * x

    def analytical_solution(self, t) -> np.ndarray:
        t_val = t.numpy() if hasattr(t, 'numpy') else t
        x0 = self.domain_config['initial_conditions']['x0']
        v0 = self.domain_config['initial_conditions']['v0']
        omega_val = self.omega.numpy()
        zeta_val = self.zeta.numpy()
        if zeta_val < 1:
            omega_d = omega_val * np.sqrt(1 - zeta_val**2)
            A = x0
            B = (v0 + zeta_val * omega_val * x0) / omega_d
            return np.exp(-zeta_val * omega_val * t_val) * (A * np.cos(omega_d * t_val) + B * np.sin(omega_d * t_val))
        else:
            return np.zeros_like(t_val)

class HeatEquation2D(PhysicsProblem):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha = tf.constant(self.domain_config['alpha'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, xyt: tf.Tensor) -> tf.Tensor:
        """
        Calcula el residual de la ecuación de calor 2D: u_t - α(∇²u) = 0
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xyt)
            u = model(xyt)
            u_x = tape.gradient(u, xyt)[:, 0:1]
            u_y = tape.gradient(u, xyt)[:, 1:2]
            u_t = tape.gradient(u, xyt)[:, 2:3]
        
        u_xx = tape.gradient(u_x, xyt)[:, 0:1]
        u_yy = tape.gradient(u_y, xyt)[:, 1:2]
        del tape

        if u_xx is None or u_yy is None or u_t is None:
            raise ValueError("El cálculo de derivadas para HEAT falló.")

        return u_t - self.alpha * (u_xx + u_yy)

    def analytical_solution(self, xyt) -> np.ndarray:
        """
        Solución analítica para la ecuación de calor 2D con condiciones de contorno cero.
        """
        xyt_val = xyt.numpy() if hasattr(xyt, 'numpy') else xyt
        x = xyt_val[:, 0:1]
        y = xyt_val[:, 1:2] 
        t = xyt_val[:, 2:3]
        alpha_val = self.alpha.numpy()
        
        return np.exp(-alpha_val * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def _extract_data_with_mapping(self) -> Optional[Tuple]:
        """Extract data using column mapping for Heat Equation"""
        x_col = self.column_mapping.get('x')
        y_col = self.column_mapping.get('y')
        time_col = self.column_mapping.get('time')
        temp_col = self.column_mapping.get('temperature')
        
        if not all([x_col, y_col, time_col, temp_col]):
            raise ValueError("Column mapping missing for HEAT. Required: 'x', 'y', 'time', 'temperature'")
        
        x_data = self.csv_data[x_col].values.reshape(-1, 1)
        y_data = self.csv_data[y_col].values.reshape(-1, 1)
        t_data = self.csv_data[time_col].values.reshape(-1, 1)
        u_data = self.csv_data[temp_col].values.reshape(-1, 1)
        
        return x_data, y_data, t_data, u_data

# --- Fábrica de Problemas Físicos ---
PROBLEMS: Dict[str, type] = {
    "SHO": SimpleHarmonicOscillator,
    "DHO": DampedHarmonicOscillator,
    "HEAT": HeatEquation2D
}

def get_physics_problem(problem_name: str, config: Dict[str, Any], 
                       csv_data: Optional[pd.DataFrame] = None, 
                       column_mapping: Optional[Dict[str, str]] = None) -> PhysicsProblem:
    """
    Fábrica de problemas físicos que instancia un problema basado en su nombre.
    """
    problem_name = problem_name.upper()
    if problem_name not in PROBLEMS:
        valid_problems = list(PROBLEMS.keys())
        raise ValueError(f"Problema '{problem_name}' no reconocido. Opciones: {valid_problems}")
    
    problem_class = PROBLEMS[problem_name]
    problem = problem_class(config)
    
    # Set CSV data with column mapping if provided
    if csv_data is not None and column_mapping is not None:
        problem.set_csv_data(csv_data, column_mapping)
    
    return problem