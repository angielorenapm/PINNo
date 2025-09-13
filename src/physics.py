# src/physics.py

"""
Módulo que define los problemas físicos y sus ecuaciones diferenciales.

Cada clase representa un problema físico específico y contiene la lógica para
calcular el residuo de su ecuación diferencial usando diferenciación automática
con TensorFlow.
"""

'''''
Importante para el training:
physics.py (Define la Regla Física)

    El trabajo de este archivo es crear un método como pde_residual().
    Este método toma el modelo y unos puntos, y devuelve un tensor con los residuos de la ecuación. 
    Es decir, devuelve un conjunto de números que deberían ser cero si el modelo fuera perfecto.
    En resumen: physics.py le entrega al Trainer un reporte de "errores físicos".

training.py (Calcula el Costo)

    El Trainer recibirá un objeto creado a partir de una clase de physics.py (por ejemplo, un objeto SimpleHarmonicOscillator).
    Dentro de su bucle de entrenamiento, el Trainer llamará al método physics_object.pde_residual().
    Luego, el Trainer tomará ese "reporte de errores" (el tensor de residuos) y lo convertirá en un solo número: la pérdida (o costo). Típicamente, lo hace calculando el error cuadrático medio.

    # Esta lógica vive dentro del Trainer en training.py
residual = self.physics_problem.pde_residual(self.model, points)
loss_pde = tf.reduce_mean(tf.square(residual)) # <--- Aquí se calcula el costo

'''''

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any

# --- Clase Base Abstracta para un Problema Físico ---

class PhysicsProblem(ABC):
    """
    Clase base abstracta para cualquier problema físico.
    Define la interfaz que todos los problemas deben implementar.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_config = config['PHYSICS_CONFIG']

    @abstractmethod
    def pde_residual(self, model: tf.keras.Model, points: tf.Tensor):
        """
        Calcula el residuo de la ecuación diferencial (la parte que debe ser cero).
        Este es el método principal que define la física del problema.
        """
        pass

# --- Implementaciones de Problemas Físicos Específicos ---

class SimpleHarmonicOscillator(PhysicsProblem):
    """
    Define el Oscilador Armónico Simple (SHO).
    Ecuación: d²x/dt² + ω²x = 0
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.omega = tf.constant(self.domain_config['omega'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, t: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            # Primera derivada: dx/dt
            x_t = tape.gradient(x, t)
        # Segunda derivada: d²x/dt²
        x_tt = tape.gradient(x_t, t)
        del tape

        # Residuo de la EDO
        residual = x_tt + (self.omega**2) * x
        return residual

class DampedHarmonicOscillator(PhysicsProblem):
    """
    Define el Oscilador Armónico Amortiguado (DHO).
    Ecuación: d²x/dt² + 2ζω(dx/dt) + ω²x = 0
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.omega = tf.constant(self.domain_config['omega'], dtype=tf.float32)
        self.zeta = tf.constant(self.domain_config['zeta'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, t: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            # Primera derivada: dx/dt
            x_t = tape.gradient(x, t)
        # Segunda derivada: d²x/dt²
        x_tt = tape.gradient(x_t, t)
        del tape

        # Residuo de la EDO
        residual = x_tt + 2 * self.zeta * self.omega * x_t + (self.omega**2) * x
        return residual

class WaveEquation1D(PhysicsProblem):
    """
    Define la Ecuación de Onda 1D.
    Ecuación: ∂²u/∂t² = c² * ∂²u/∂x²
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.c = tf.constant(self.domain_config['c'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, xt: tf.Tensor) -> tf.Tensor:
        x, t = xt[:, 0:1], xt[:, 1:2]
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            u = model(tf.concat([x, t], axis=1))
            
            # Derivadas primeras
            u_x = tape.gradient(u, x)
            u_t = tape.gradient(u, t)
        
        # Derivadas segundas
        u_xx = tape.gradient(u_x, x)
        u_tt = tape.gradient(u_t, t)
        del tape

        # Residuo de la EDP
        residual = u_tt - (self.c**2) * u_xx
        return residual

# --- Fábrica de Problemas Físicos ---

# Mapeo de strings a clases de problemas
PROBLEMS: Dict[str, type] = {
    "SHO": SimpleHarmonicOscillator,
    "DHO": DampedHarmonicOscillator,
    "WAVE": WaveEquation1D
}

def get_physics_problem(problem_name: str, config: Dict[str, Any]) -> PhysicsProblem:
    """
    Fábrica que instancia el problema físico correcto a partir de su nombre.

    Args:
        problem_name (str): El nombre del problema a resolver (ej. "SHO").
        config (Dict[str, Any]): El diccionario de configuración completo.

    Returns:
        PhysicsProblem: Una instancia de la clase del problema físico solicitado.
    """
    problem_name = problem_name.upper()
    if problem_name not in PROBLEMS:
        raise ValueError(f"Problema '{problem_name}' no reconocido. Opciones: {list(PROBLEMS.keys())}")
    
    problem_class = PROBLEMS[problem_name]
    
    # Pasa la configuración completa a la clase del problema
    # ya que `config.py` ahora contiene todas las configs necesarias
    return problem_class(
        {"PHYSICS_CONFIG": config["PHYSICS_CONFIG"]}
    )