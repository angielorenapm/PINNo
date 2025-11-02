# src/losses.py
"""
Módulo para cálculo de funciones de pérdida en PINNs.

Responsabilidades:
- Definir todas las funciones de pérdida específicas por problema
- Calcular pérdidas compuestas con pesos
- Proporcionar interfaz consistente para el trainer

Pattern: Strategy Pattern para diferentes tipos de pérdida
"""
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List


class LossCalculator:
    """
    Calculador de pérdidas para PINNs.
    
    Pattern: Strategy + Composite
    """
    
    def __init__(self, loss_weights: Dict[str, float], problem_name: str):
        self.loss_weights = loss_weights
        self.problem_name = problem_name
        
        # Mapeo de funciones de pérdida por problema
        self.loss_functions = {
            "SHO": self._compute_sho_losses,
            "DHO": self._compute_sho_losses,  # Misma estructura que SHO
            "WAVE": self._compute_wave_losses
        }

    def compute_losses(self, model: tf.keras.Model, physics, 
                      training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Calcula todas las pérdidas para el problema actual.
        
        Args:
            model: Modelo de red neuronal
            physics: Problema físico
            training_data: Diccionario con datos de entrenamiento
            
        Returns:
            Tuple: (pérdida_total, lista_de_pérdidas_componentes)
        """
        loss_fn = self.loss_functions.get(self.problem_name)
        if not loss_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        return loss_fn(model, physics, training_data)

    def _compute_sho_losses(self, model: tf.keras.Model, physics, 
                           training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Calcula pérdidas para osciladores armónicos"""
        t_coll, t0, x0_true, v0_true = self._extract_sho_data(training_data)
        
        # Pérdida de PDE
        residual = physics.pde_residual(model, t_coll)
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # Pérdida inicial
        loss_initial = self._initial_loss_sho(model, t0, x0_true, v0_true)
        
        # Pérdida total ponderada
        total_loss = (self.loss_weights["ode"] * loss_pde + 
                     self.loss_weights["initial"] * loss_initial)
        
        return total_loss, [loss_pde, loss_initial]

    def _compute_wave_losses(self, model: tf.keras.Model, physics,
                            training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Calcula pérdidas para ecuación de onda"""
        xt_coll, xt0, xt_b = self._extract_wave_data(training_data)
        
        # Pérdida de PDE
        residual = physics.pde_residual(model, xt_coll)
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # Pérdidas de condiciones iniciales y de contorno
        loss_initial = self._initial_loss_wave(model, xt0)
        loss_boundary = self._boundary_loss_wave(model, xt_b)
        
        # Pérdida total ponderada
        total_loss = (self.loss_weights["pde"] * loss_pde +
                     self.loss_weights["initial"] * loss_initial +
                     self.loss_weights["boundary"] * loss_boundary)
        
        return total_loss, [loss_pde, loss_initial, loss_boundary]

    def _extract_sho_data(self, training_data: Dict[str, tf.Tensor]):
        """Extrae datos para problemas SHO/DHO"""
        return (training_data["t_coll"], training_data["t0"], 
                training_data["x0_true"], training_data["v0_true"])

    def _extract_wave_data(self, training_data: Dict[str, tf.Tensor]):
        """Extrae datos para problemas de onda"""
        return (training_data["xt_coll"], training_data["xt0"], 
                training_data["xt_b"])

    # Funciones de pérdida específicas (mantenidas de la versión original)
    def _initial_loss_sho(self, model: tf.keras.Model, t0: tf.Tensor, 
                         x0_true: float, v0_true: float) -> tf.Tensor:
        """Pérdida para condiciones iniciales de SHO"""
        with tf.GradientTape() as tape:
            tape.watch(t0)
            x0_pred = model(t0)
        v0_pred = tape.gradient(x0_pred, t0)
        
        loss_x0 = tf.reduce_mean(tf.square(x0_pred - x0_true))
        loss_v0 = tf.reduce_mean(tf.square(v0_pred - v0_true))
        
        return loss_x0 + loss_v0

    def _initial_loss_wave(self, model: tf.keras.Model, xt0: tf.Tensor) -> tf.Tensor:
        """Pérdida para condiciones iniciales de onda"""
        x = xt0[:, 0:1]
        u_pred = model(xt0)
        u_true = tf.sin(np.pi * x)
        loss_u = tf.reduce_mean(tf.square(u_pred - u_true))
        
        with tf.GradientTape() as tape:
            tape.watch(xt0)
            u = model(xt0)
        u_t = tape.gradient(u, xt0)[:, 1:2]
        loss_ut = tf.reduce_mean(tf.square(u_t))
        
        return loss_u + loss_ut

    def _boundary_loss_wave(self, model: tf.keras.Model, xt_b: tf.Tensor) -> tf.Tensor:
        """Pérdida para condiciones de contorno de onda"""
        u_b_pred = model(xt_b)
        return tf.reduce_mean(tf.square(u_b_pred))