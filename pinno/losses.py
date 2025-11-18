"""
Módulo para cálculo de funciones de pérdida en PINNs.

Este módulo implementa el patrón Strategy para definir y seleccionar diferentes
funciones de coste según el problema físico (SHO, HEAT, etc.). Calcula tanto
el residual de la ecuación diferencial (pérdida física) como el error respecto
a datos observados.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List


class LossCalculator:
    """
    Calculador de pérdidas compuesto para entrenamiento de PINNs.

    Esta clase orquesta el cálculo de la función de coste total, combinando:
    - Pérdida de la PDE (Residual físico).
    - Pérdida de condiciones iniciales y de frontera.
    - Pérdida de datos observados (si existen).

    Attributes:
        loss_weights (Dict[str, float]): Pesos para ponderar cada componente del error.
        problem_name (str): Identificador del problema físico actual.
        loss_functions (dict): Mapeo interno de estrategias de pérdida.
    """
    
    def __init__(self, loss_weights: Dict[str, float], problem_name: str):
        """
        Inicializa el calculador de pérdidas.

        Args:
            loss_weights (Dict[str, float]): Diccionario con pesos (ej. {"ode": 1.0, "data": 10.0}).
            problem_name (str): Nombre del problema ("SHO", "HEAT", etc.).
        """
        self.loss_weights = loss_weights
        self.problem_name = problem_name
        
        # Mapeo de funciones de pérdida por problema
        self.loss_functions = {
            "SHO": self._compute_sho_losses,
            "DHO": self._compute_sho_losses,  # DHO usa la misma estructura que SHO
            "HEAT": self._compute_heat_losses
        }

    def compute_losses(self, model: tf.keras.Model, physics, 
                      training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Ejecuta el cálculo de pérdida correspondiente al problema configurado.

        Args:
            model (tf.keras.Model): La red neuronal (PINN) a evaluar.
            physics: Instancia del problema físico con métodos `pde_residual`.
            training_data (Dict[str, tf.Tensor]): Datos de entrada (colocación, iniciales, externos).

        Returns:
            Tuple[tf.Tensor, List[tf.Tensor]]:
                - loss total (escalar): Suma ponderada para el optimizador.
                - componentes (lista): Valores individuales de cada pérdida para monitoreo.

        Raises:
            ValueError: Si el problema configurado no tiene una función de pérdida definida.
        """
        loss_fn = self.loss_functions.get(self.problem_name)
        if not loss_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        return loss_fn(model, physics, training_data)

    def _compute_sho_losses(self, model: tf.keras.Model, physics, 
                           training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Calcula pérdidas para osciladores armónicos (SHO/DHO).

        Componentes:
        1. Residual de la ODE (Física).
        2. Condición inicial (x0, v0).
        3. Datos externos (si `ext_t` y `ext_x` están presentes).
        """
        # 1. Pérdida Física (PDE)
        # Calculamos qué tan lejos está la red de cumplir la ecuación diferencial
        residual = physics.pde_residual(model, training_data["t_coll"])
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # 2. Pérdida Inicial
        # Verificamos si la red empieza en la posición y velocidad correctas
        loss_initial = self._initial_loss_sho(
            model, 
            training_data["t0"], 
            training_data["x0_true"], 
            training_data["v0_true"]
        )
        
        # 3. Pérdida de Datos Externos (Opcional)
        loss_data = tf.constant(0.0, dtype=tf.float32)
        if "ext_t" in training_data and "ext_x" in training_data:
            pred_ext = model(training_data["ext_t"])
            loss_data = tf.reduce_mean(tf.square(pred_ext - training_data["ext_x"]))

        # Suma Ponderada
        w_ode = self.loss_weights.get("ode", 1.0)
        w_init = self.loss_weights.get("initial", 1.0)
        w_data = self.loss_weights.get("data", 10.0) # Peso alto por defecto para datos

        total_loss = (w_ode * loss_pde + w_init * loss_initial + w_data * loss_data)
        
        return total_loss, [loss_pde, loss_initial, loss_data]

    def _compute_heat_losses(self, model: tf.keras.Model, physics,
                            training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Calcula pérdidas para la ecuación de calor 2D.

        Componentes:
        1. Residual de la PDE (Calor).
        2. Condiciones iniciales (t=0) y de frontera (bordes espaciales).
        3. Datos externos (si `ext_xyt` y `ext_u` están presentes).
        """
        # 1. PDE
        residual = physics.pde_residual(model, training_data["xyt_coll"])
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # 2. Condiciones Iniciales y Frontera
        u_pred_0 = model(training_data["xyt0"])
        u_true_0 = physics.analytical_solution(training_data["xyt0"])
        loss_initial = tf.reduce_mean(tf.square(u_pred_0 - u_true_0))

        u_pred_b = model(training_data["xyt_b"])
        u_true_b = physics.analytical_solution(training_data["xyt_b"])
        loss_boundary = tf.reduce_mean(tf.square(u_pred_b - u_true_b))
        
        # 3. Pérdida de Datos Externos (Opcional)
        loss_data = tf.constant(0.0, dtype=tf.float32)
        if "ext_xyt" in training_data and "ext_u" in training_data:
            pred_ext = model(training_data["ext_xyt"])
            loss_data = tf.reduce_mean(tf.square(pred_ext - training_data["ext_u"]))

        # Pesos
        w_pde = self.loss_weights.get("pde", 1.0)
        w_init = self.loss_weights.get("initial", 1.0)
        w_bound = self.loss_weights.get("boundary", 1.0)
        w_data = self.loss_weights.get("data", 10.0)

        total_loss = (w_pde * loss_pde + w_init * loss_initial + 
                      w_bound * loss_boundary + w_data * loss_data)
                      
        return total_loss, [loss_pde, loss_initial, loss_boundary, loss_data]

    def _initial_loss_sho(self, model: tf.keras.Model, t0: tf.Tensor, 
                         x0_true: float, v0_true: float) -> tf.Tensor:
        """
        Calcula el error en las condiciones iniciales para SHO (posición y velocidad).
        
        Usa diferenciación automática para calcular la velocidad predicha (v = dx/dt).
        """
        with tf.GradientTape() as tape:
            tape.watch(t0)
            x0_pred = model(t0)
        # Calcular velocidad como la derivada de la posición respecto al tiempo
        v0_pred = tape.gradient(x0_pred, t0)
        
        loss_x = tf.reduce_mean(tf.square(x0_pred - x0_true))
        loss_v = tf.reduce_mean(tf.square(v0_pred - v0_true))
        
        return loss_x + loss_v