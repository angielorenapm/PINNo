# pinno/losses.py
"""
Módulo para cálculo de funciones de pérdida (Loss Functions) en PINNs.

Este módulo implementa el patrón Strategy para seleccionar la función de coste 
adecuada según el problema físico. Calcula tanto el residual físico (PDE) 
como el error respecto a datos observados (si existen).
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class LossCalculator:
    """
    Calculador de pérdidas compuesto para entrenamiento de PINNs.
    
    Orquesta el cálculo del error total combinando:
    - Pérdida de la PDE (Residual físico).
    - Pérdida de condiciones iniciales/frontera (Analítico).
    - Pérdida de datos observados (Data-Driven).
    """
    
    def __init__(self, loss_weights: Dict[str, float], problem_name: str):
        """
        Inicializa el calculador con pesos y el problema activo.

        Args:
            loss_weights (Dict[str, float]): Pesos para ponderar componentes del error 
                                             (ej. ``{"ode": 1.0, "data": 100.0}``).
            problem_name (str): Identificador del problema ("SHO", "HEAT").
        """
        self.loss_weights = loss_weights
        self.problem_name = problem_name
        
        self.loss_functions = {
            "SHO": self._compute_sho_losses,
            "DHO": self._compute_sho_losses,
            "HEAT": self._compute_heat_losses
        }

    def compute_losses(self, model: tf.keras.Model, physics, 
                      training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Ejecuta el cálculo de pérdida correspondiente al problema configurado.

        Args:
            model (tf.keras.Model): La red neuronal (PINN) a evaluar.
            physics: Instancia del problema físico con métodos ``pde_residual``.
            training_data (Dict[str, tf.Tensor]): Diccionario con tensores de entrada.

        Raises:
            ValueError: Si el problema configurado no tiene función de pérdida asociada.

        Returns:
            Tuple[tf.Tensor, List[tf.Tensor]]:
                - **Total Loss**: Escalar (suma ponderada) para el optimizador.
                - **Components**: Lista de valores individuales de cada pérdida.
        """
        loss_fn = self.loss_functions.get(self.problem_name)
        if not loss_fn: 
            raise ValueError(f"Problema {self.problem_name} no soportado")
        return loss_fn(model, physics, training_data)

    def _compute_sho_losses(self, model: tf.keras.Model, physics, 
                           training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Calcula pérdidas para osciladores armónicos (SHO/DHO).
        
        Detecta automáticamente el modo (Analítico vs CSV) inspeccionando las claves de datos.

        Args:
            model: Red Neuronal.
            physics: Objeto físico.
            training_data: Datos de entrada.

        Returns:
            Tuple: (Loss Total, [Loss PDE, Loss Data/Initial]).
        """
        # --- FIX: Deteccion automatica por claves de datos ---
        # Si "t0" esta en los datos, estamos en modo Analitico. Si no, CSV.
        if "t0" in training_data:
            # Modo Analitico
            residual = physics.pde_residual(model, training_data["t_coll"])
            loss_pde = tf.reduce_mean(tf.square(residual))
            
            loss_initial = self._initial_loss_sho(
                model, training_data["t0"], 
                training_data["x0_true"], training_data["v0_true"]
            )
            
            total = (self.loss_weights.get("ode", 1.0) * loss_pde + 
                     self.loss_weights.get("initial", 100.0) * loss_initial)
            return total, [loss_pde, loss_initial]
        else:
            # Modo CSV (Data Driven)
            residual = physics.pde_residual(model, training_data["t_coll"])
            loss_pde = tf.reduce_mean(tf.square(residual))
            
            # Ajuste a datos reales
            x_pred = model(training_data["t_data"])
            loss_data = tf.reduce_mean(tf.square(x_pred - training_data["x_data"]))
            
            total = (self.loss_weights.get("ode", 0.1) * loss_pde + 
                     self.loss_weights.get("data", 200.0) * loss_data)
            return total, [loss_pde, loss_data]

    def _compute_heat_losses(self, model: tf.keras.Model, physics, 
                            training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Calcula pérdidas para la ecuación de calor 2D.

        Args:
            model: Red Neuronal.
            physics: Objeto físico (HeatEquation2D).
            training_data: Datos de entrada (xyt).

        Returns:
            Tuple: (Loss Total, [Loss PDE, Loss Init, Loss Bound] o [Loss PDE, Loss Data]).
        """
        # Mismo fix para HEAT: detectar 'xyt0'
        if "xyt0" in training_data:
            # Analitico
            residual = physics.pde_residual(model, training_data["xyt_coll"])
            loss_pde = tf.reduce_mean(tf.square(residual))
            
            u0 = model(training_data["xyt0"])
            loss_init = tf.reduce_mean(tf.square(u0 - physics.analytical_solution(training_data["xyt0"])))
            
            ub = model(training_data["xyt_b"])
            loss_bound = tf.reduce_mean(tf.square(ub - physics.analytical_solution(training_data["xyt_b"])))
            
            total = (self.loss_weights.get("pde", 1.0)*loss_pde + 
                     self.loss_weights.get("initial", 100.0)*loss_init + 
                     self.loss_weights.get("boundary", 100.0)*loss_bound)
            return total, [loss_pde, loss_init, loss_bound]
        else:
            # CSV
            residual = physics.pde_residual(model, training_data["xyt_coll"])
            loss_pde = tf.reduce_mean(tf.square(residual))
            
            # Construir input (x, y, t) desde datos CSV
            xyt_data = tf.concat([training_data["x_data"], 
                                  training_data["y_data"], 
                                  training_data["t_data"]], axis=1)
            u_pred = model(xyt_data)
            loss_data = tf.reduce_mean(tf.square(u_pred - training_data["u_data"]))
            
            total = self.loss_weights.get("pde", 0.1) * loss_pde + self.loss_weights.get("data", 200.0) * loss_data
            return total, [loss_pde, loss_data]

    def _initial_loss_sho(self, model: tf.keras.Model, t0: tf.Tensor, 
                         x0_true: tf.Tensor, v0_true: tf.Tensor) -> tf.Tensor:
        """
        Calcula el error de condición inicial para osciladores (posición y velocidad).

        Args:
            model: Red Neuronal.
            t0 (tf.Tensor): Tiempo inicial (0.0).
            x0_true (tf.Tensor): Posición inicial verdadera.
            v0_true (tf.Tensor): Velocidad inicial verdadera.

        Returns:
            tf.Tensor: Suma del error cuadrático de posición y velocidad.
        """
        with tf.GradientTape() as tape:
            tape.watch(t0)
            x0_pred = model(t0)
        v0_pred = tape.gradient(x0_pred, t0)
        if v0_pred is None: v0_pred = tf.zeros_like(t0)
        return tf.reduce_mean(tf.square(x0_pred - x0_true)) + tf.reduce_mean(tf.square(v0_pred - v0_true))