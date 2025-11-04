# src/physics.py

"""
Módulo que define los problemas físicos y sus ecuaciones diferenciales.
(Versión 0.0.3 - Con ecuación de calor 2D)
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

class HeatEquation2D(PhysicsProblem):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha = tf.constant(self.domain_config['alpha'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, xyt: tf.Tensor) -> tf.Tensor:
        """
        Calcula el residual de la ecuación de calor 2D: u_t - α(∇²u) = 0
        
        Args:
            model: Modelo de red neuronal
            xyt: Tensor de forma (batch, 3) con coordenadas [x, y, t]
            
        Returns:
            Tensor: Residual de la PDE
        """
        # xyt: tensor de forma (batch, 3) donde [x, y, t]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xyt)
            u = model(xyt)
            # Primeras derivadas
            u_x = tape.gradient(u, xyt)[:, 0:1]
            u_y = tape.gradient(u, xyt)[:, 1:2]
            u_t = tape.gradient(u, xyt)[:, 2:3]
        
        # Segundas derivadas
        u_xx = tape.gradient(u_x, xyt)[:, 0:1]
        u_yy = tape.gradient(u_y, xyt)[:, 1:2]
        del tape

        # Verificación de que las derivadas se calcularon correctamente
        if u_xx is None or u_yy is None or u_t is None:
            raise ValueError("El cálculo de derivadas para HEAT falló.")

        # Ecuación de calor: u_t - alpha * (u_xx + u_yy) = 0
        return u_t - self.alpha * (u_xx + u_yy)

    def analytical_solution(self, xyt) -> np.ndarray:
        """
        Solución analítica para la ecuación de calor 2D con condiciones de contorno cero.
        
        Para una placa cuadrada [0,1]×[0,1] con condición inicial:
        u(x,y,0) = sin(πx) * sin(πy)
        
        La solución es:
        u(x,y,t) = exp(-απ²t) * sin(πx) * sin(πy)
        """
        xyt_val = xyt.numpy() if hasattr(xyt, 'numpy') else xyt
        x = xyt_val[:, 0:1]
        y = xyt_val[:, 1:2] 
        t = xyt_val[:, 2:3]
        alpha_val = self.alpha.numpy()
        
        # Solución fundamental para condiciones de contorno cero
        return np.exp(-alpha_val * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

# --- Fábrica de Problemas Físicos ---
PROBLEMS: Dict[str, type] = {
    "SHO": SimpleHarmonicOscillator,
    "DHO": DampedHarmonicOscillator,
    "HEAT": HeatEquation2D  # Cambiado de "WAVE" a "HEAT"
}

def get_physics_problem(problem_name: str, config: Dict[str, Any]) -> PhysicsProblem:
    """
    Fábrica de problemas físicos que instancia un problema basado en su nombre.
    
    Args:
        problem_name: Nombre del problema ("SHO", "DHO", "HEAT")
        config: Configuración del problema
        
    Returns:
        Instancia del problema físico
        
    Raises:
        ValueError: Si el nombre del problema no es reconocido
    """
    problem_name = problem_name.upper()
    if problem_name not in PROBLEMS:
        valid_problems = list(PROBLEMS.keys())
        raise ValueError(f"Problema '{problem_name}' no reconocido. Opciones: {valid_problems}")
    problem_class = PROBLEMS[problem_name]
    return problem_class(config)