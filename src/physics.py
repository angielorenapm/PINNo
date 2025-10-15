# src/physics.py

"""
Módulo que define los problemas físicos y sus ecuaciones diferenciales.
(Versión final con el cálculo de gradientes corregido)
"""
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any

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
        # ¡CORRECCIÓN! El cálculo de la primera derivada DEBE estar dentro del 'with'.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            x_t = tape.gradient(x, t)
        # La segunda derivada se calcula después, usando la cinta que grabó todo.
        x_tt = tape.gradient(x_t, t)
        del tape
        
        # Comprobación para evitar el error 'None'
        if x_tt is None:
            raise ValueError("El cálculo de la segunda derivada (x_tt) falló y devolvió None. "
                             "Asegúrate de que el modelo sea diferenciable.")
                             
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
        # ¡CORRECCIÓN! Misma lógica que en SHO.
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

class WaveEquation1D(PhysicsProblem):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.c = tf.constant(self.domain_config['c'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, xt: tf.Tensor) -> tf.Tensor:
        # ¡CORRECCIÓN! Misma lógica.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xt)
            u = model(xt)
            u_grads = tape.gradient(u, xt)
        
        # El gradiente de un gradiente requiere un paso intermedio
        u_t_grads = tape.gradient(u_grads, xt)
        del tape

        u_tt = u_t_grads[:, 1:2]
        u_xx = u_t_grads[:, 0:1] # Esto es incorrecto, se debe derivar u_x respecto a x
        
        # Vamos a hacerlo de la forma más clara y segura
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xt)
            u = model(xt)
            u_x = tape.gradient(u, xt)[:, 0:1]
            u_t = tape.gradient(u, xt)[:, 1:2]
        u_xx = tape.gradient(u_x, xt)[:, 0:1]
        u_tt = tape.gradient(u_t, xt)[:, 1:2]
        del tape

        if u_xx is None or u_tt is None:
             raise ValueError("El cálculo de derivadas para WAVE falló.")

        return u_tt - (self.c**2) * u_xx

    def analytical_solution(self, xt) -> np.ndarray:
        xt_val = xt.numpy() if hasattr(xt, 'numpy') else xt
        c_val = self.c.numpy()
        x = xt_val[:, 0:1]
        t = xt_val[:, 1:2]
        return np.sin(np.pi * x) * np.cos(c_val * np.pi * t)

# --- Fábrica de Problemas Físicos (sin cambios) ---
PROBLEMS: Dict[str, type] = {
    "SHO": SimpleHarmonicOscillator,
    "DHO": DampedHarmonicOscillator,
    "WAVE": WaveEquation1D
}

def get_physics_problem(problem_name: str, config: Dict[str, Any]) -> PhysicsProblem:
    problem_name = problem_name.upper()
    if problem_name not in PROBLEMS:
        raise ValueError(f"Problema '{problem_name}' no reconocido. Opciones: {list(PROBLEMS.keys())}")
    problem_class = PROBLEMS[problem_name]
    return problem_class(config)