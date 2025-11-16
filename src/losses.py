"""
Módulo para cálculo de funciones de pérdida en PINNs.
(Versión 0.0.4 - Con soporte para datos CSV)
"""
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class LossCalculator:
    """
    Calculador de pérdidas para PINNs con soporte para datos CSV.
    """
    
    def __init__(self, loss_weights: Dict[str, float], problem_name: str):
        self.loss_weights = loss_weights
        self.problem_name = problem_name
        
        # Mapeo de funciones de pérdida por problema
        self.loss_functions = {
            "SHO": self._compute_sho_losses,
            "DHO": self._compute_sho_losses,
            "HEAT": self._compute_heat_losses
        }

    def compute_losses(self, model: tf.keras.Model, physics, 
                      training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Calcula todas las pérdidas para el problema actual.
        """
        loss_fn = self.loss_functions.get(self.problem_name)
        if not loss_fn:
            raise ValueError(f"Problema no soportado: {self.problem_name}")
            
        return loss_fn(model, physics, training_data)

    def _compute_sho_losses(self, model: tf.keras.Model, physics, 
                           training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Calcula pérdidas para osciladores armónicos"""
        if physics.has_analytical:
            # Modo analítico
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
        else:
            # Modo data-driven con CSV
            return self._compute_sho_csv_losses(model, physics, training_data)

    def _compute_heat_losses(self, model: tf.keras.Model, physics,
                            training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Calcula pérdidas para la ecuación de calor 2D"""
        if physics.has_analytical:
            # Modo analítico
            xyt_coll, xyt0, xyt_b = self._extract_heat_data(training_data)
            
            # Pérdida de PDE
            residual = physics.pde_residual(model, xyt_coll)
            loss_pde = tf.reduce_mean(tf.square(residual))
            
            # Pérdidas de condiciones iniciales y de contorno
            loss_initial = self._initial_loss_heat(model, xyt0, physics)
            loss_boundary = self._boundary_loss_heat(model, xyt_b, physics)
            
            # Pérdida total ponderada
            total_loss = (self.loss_weights["pde"] * loss_pde +
                         self.loss_weights["initial"] * loss_initial +
                         self.loss_weights["boundary"] * loss_boundary)
            
            return total_loss, [loss_pde, loss_initial, loss_boundary]
        else:
            # Modo data-driven con CSV
            return self._compute_heat_csv_losses(model, physics, training_data)

    def _compute_sho_csv_losses(self, model: tf.keras.Model, physics,
                               training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Calcula pérdidas para SHO/DHO con datos CSV"""
        t_coll = training_data["t_coll"]
        t_data = training_data["t_data"]
        x_data = training_data["x_data"]
        
        # Pérdida de PDE
        residual = physics.pde_residual(model, t_coll)
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # Pérdida de datos
        x_pred = model(t_data)
        loss_data = tf.reduce_mean(tf.square(x_pred - x_data))
        
        # Pérdida total ponderada
        total_loss = (self.loss_weights["ode"] * loss_pde +
                     self.loss_weights.get("data", 10.0) * loss_data)
        
        return total_loss, [loss_pde, loss_data]

    def _compute_heat_csv_losses(self, model: tf.keras.Model, physics,
                                training_data: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Calcula pérdidas para Heat Equation con datos CSV"""
        xyt_coll = training_data["xyt_coll"]
        x_data = training_data["x_data"]
        y_data = training_data["y_data"]
        t_data = training_data["t_data"]
        u_data = training_data["u_data"]
        
        # Pérdida de PDE
        residual = physics.pde_residual(model, xyt_coll)
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # Pérdida de datos
        xyt_data = tf.concat([x_data, y_data, t_data], axis=1)
        u_pred = model(xyt_data)
        loss_data = tf.reduce_mean(tf.square(u_pred - u_data))
        
        # Pérdida total ponderada
        total_loss = (self.loss_weights["pde"] * loss_pde +
                     self.loss_weights.get("data", 10.0) * loss_data)
        
        return total_loss, [loss_pde, loss_data]

    def _extract_sho_data(self, training_data: Dict[str, tf.Tensor]):
        """Extrae datos para problemas SHO/DHO analíticos"""
        return (training_data["t_coll"], training_data["t0"], 
                training_data["x0_true"], training_data["v0_true"])

    def _extract_heat_data(self, training_data: Dict[str, tf.Tensor]):
        """Extrae datos para problemas de calor 2D analíticos"""
        return (training_data["xyt_coll"], training_data["xyt0"], 
                training_data["xyt_b"])

    # Funciones de pérdida específicas para SHO/DHO analíticos
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

    # Funciones de pérdida específicas para HEAT analíticos
    def _initial_loss_heat(self, model: tf.keras.Model, xyt0: tf.Tensor, physics) -> tf.Tensor:
        """Pérdida para condiciones iniciales de calor 2D"""
        u_pred = model(xyt0)
        u_true = physics.analytical_solution(xyt0)
        return tf.reduce_mean(tf.square(u_pred - u_true))

    def _boundary_loss_heat(self, model: tf.keras.Model, xyt_b: tf.Tensor, physics) -> tf.Tensor:
        """Pérdida para condiciones de contorno de calor 2D"""
        u_pred = model(xyt_b)
        u_true = physics.analytical_solution(xyt_b)
        return tf.reduce_mean(tf.square(u_pred - u_true))