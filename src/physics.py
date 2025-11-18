# src/physics.py

"""
Módulo que define los problemas físicos y sus ecuaciones diferenciales.
(Versión 0.0.4 - Con métricas de Slicing)
"""
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

# --- Clase Base Abstracta ---
class PhysicsProblem(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_config = config['PHYSICS_CONFIG']

    @abstractmethod
    def pde_residual(self, model: tf.keras.Model, points: tf.Tensor):
        pass
    
    @abstractmethod
    def analytical_solution(self, points) -> np.ndarray:
        pass

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
            raise ValueError("Error derivada segunda SHO")
                             
        return x_tt + (self.omega**2) * x
    
    def analytical_solution(self, t) -> np.ndarray:
        t_val = t.numpy() if hasattr(t, 'numpy') else t
        x0 = self.domain_config['initial_conditions']['x0']
        v0 = self.domain_config['initial_conditions']['v0']
        omega_val = self.omega.numpy()
        return x0 * np.cos(omega_val * t_val) + (v0 / omega_val) * np.sin(omega_val * t_val)

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

        if x_tt is None: raise ValueError("Error derivada DHO")

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
            raise ValueError("Error derivadas HEAT")

        return u_t - self.alpha * (u_xx + u_yy)

    def analytical_solution(self, xyt) -> np.ndarray:
        xyt_val = xyt.numpy() if hasattr(xyt, 'numpy') else xyt
        x = xyt_val[:, 0:1]
        y = xyt_val[:, 1:2] 
        t = xyt_val[:, 2:3]
        alpha_val = self.alpha.numpy()
        return np.exp(-alpha_val * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    # --- NUEVO: Cálculo de métricas por slices (Requerimiento 4) ---
    def compute_slice_metrics(self, model: tf.keras.Model, t_fix: float = 0.5, resolution: int = 50) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calcula MSE en un corte de tiempo específico (slice).
        Retorna: (mse, u_pred, u_true) para ese instante t.
        """
        x_d = self.domain_config['x_domain']
        y_d = self.domain_config['y_domain']
        
        # Crear grid espacial
        x = np.linspace(x_d[0], x_d[1], resolution)
        y = np.linspace(y_d[0], y_d[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Crear tensor input con t fijo
        T = np.full_like(X, t_fix)
        
        # Aplanar y apilar: (N, 3) -> [x, y, t]
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = T.flatten()
        
        input_array = np.stack([x_flat, y_flat, t_flat], axis=1).astype(np.float32)
        input_tensor = tf.convert_to_tensor(input_array)
        
        # Predicción
        u_pred_flat = model(input_tensor).numpy()
        u_true_flat = self.analytical_solution(input_tensor)
        
        # Calcular MSE
        mse = np.mean((u_pred_flat - u_true_flat)**2)
        
        # Reshape para visualización (50, 50)
        return mse, u_pred_flat.reshape(resolution, resolution), u_true_flat.reshape(resolution, resolution)

# --- Fábrica ---
PROBLEMS: Dict[str, type] = {
    "SHO": SimpleHarmonicOscillator,
    "DHO": DampedHarmonicOscillator,
    "HEAT": HeatEquation2D
}

def get_physics_problem(problem_name: str, config: Dict[str, Any]) -> PhysicsProblem:
    problem_name = problem_name.upper()
    if problem_name not in PROBLEMS:
        raise ValueError(f"Problema '{problem_name}' no reconocido.")
    return PROBLEMS[problem_name](config)