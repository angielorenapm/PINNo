# pinno/losses.py
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class LossCalculator:
    def __init__(self, loss_weights: Dict[str, float], problem_name: str):
        self.loss_weights = loss_weights
        self.problem_name = problem_name
        
        self.loss_functions = {
            "SHO": self._compute_sho_losses,
            "DHO": self._compute_sho_losses,
            "HEAT": self._compute_heat_losses
        }

    def compute_losses(self, model, physics, training_data):
        loss_fn = self.loss_functions.get(self.problem_name)
        if not loss_fn: raise ValueError(f"Problema {self.problem_name} no soportado")
        return loss_fn(model, physics, training_data)

    def _compute_sho_losses(self, model, physics, training_data):
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

    def _compute_heat_losses(self, model, physics, training_data):
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

    def _initial_loss_sho(self, model, t0, x0_true, v0_true):
        with tf.GradientTape() as tape:
            tape.watch(t0)
            x0_pred = model(t0)
        v0_pred = tape.gradient(x0_pred, t0)
        if v0_pred is None: v0_pred = tf.zeros_like(t0)
        return tf.reduce_mean(tf.square(x0_pred - x0_true)) + tf.reduce_mean(tf.square(v0_pred - v0_true))