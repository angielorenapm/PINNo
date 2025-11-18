# src/losses.py
import tensorflow as tf
from typing import Dict, Any, Tuple, List

class LossCalculator:
    def __init__(self, loss_weights: Dict[str, float], problem_name: str):
        self.loss_weights = loss_weights
        self.problem_name = problem_name
        
        self.loss_functions = {
            "SHO": self._compute_sho_losses,
            "DHO": self._compute_sho_losses,
            "HEAT": self._compute_heat_losses
        }

    def compute_losses(self, model, physics, training_data) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        loss_fn = self.loss_functions.get(self.problem_name)
        if not loss_fn: raise ValueError(f"Unknown problem: {self.problem_name}")
        return loss_fn(model, physics, training_data)

    def _compute_sho_losses(self, model, physics, data):
        # 1. Pérdida Física (PDE)
        residual = physics.pde_residual(model, data["t_coll"])
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # 2. Pérdida Inicial
        loss_initial = self._initial_loss_sho(model, data["t0"], data["x0_true"], data["v0_true"])
        
        # 3. NUEVO: Pérdida de Datos Externos (si existen)
        loss_data = tf.constant(0.0, dtype=tf.float32)
        if "ext_t" in data and "ext_x" in data:
            pred_ext = model(data["ext_t"])
            loss_data = tf.reduce_mean(tf.square(pred_ext - data["ext_x"]))

        # Suma Ponderada (Asignamos peso 10.0 a datos por defecto si no está en config)
        w_ode = self.loss_weights.get("ode", 1.0)
        w_init = self.loss_weights.get("initial", 1.0)
        w_data = self.loss_weights.get("data", 10.0)

        total_loss = (w_ode * loss_pde + w_init * loss_initial + w_data * loss_data)
        
        return total_loss, [loss_pde, loss_initial, loss_data]

    def _compute_heat_losses(self, model, physics, data):
        # 1. PDE
        residual = physics.pde_residual(model, data["xyt_coll"])
        loss_pde = tf.reduce_mean(tf.square(residual))
        
        # 2. Condiciones Iniciales y Frontera
        u_pred_0 = model(data["xyt0"])
        u_true_0 = physics.analytical_solution(data["xyt0"])
        loss_initial = tf.reduce_mean(tf.square(u_pred_0 - u_true_0))

        u_pred_b = model(data["xyt_b"])
        u_true_b = physics.analytical_solution(data["xyt_b"])
        loss_boundary = tf.reduce_mean(tf.square(u_pred_b - u_true_b))
        
        # 3. NUEVO: Pérdida de Datos Externos
        loss_data = tf.constant(0.0, dtype=tf.float32)
        if "ext_xyt" in data and "ext_u" in data:
            pred_ext = model(data["ext_xyt"])
            loss_data = tf.reduce_mean(tf.square(pred_ext - data["ext_u"]))

        # Pesos
        w_pde = self.loss_weights.get("pde", 1.0)
        w_init = self.loss_weights.get("initial", 1.0)
        w_bound = self.loss_weights.get("boundary", 1.0)
        w_data = self.loss_weights.get("data", 10.0)

        total_loss = (w_pde * loss_pde + w_init * loss_initial + 
                      w_bound * loss_boundary + w_data * loss_data)
                      
        return total_loss, [loss_pde, loss_initial, loss_boundary, loss_data]

    def _initial_loss_sho(self, model, t0, x0_true, v0_true):
        with tf.GradientTape() as tape:
            tape.watch(t0)
            x0_pred = model(t0)
        v0_pred = tape.gradient(x0_pred, t0)
        return tf.reduce_mean(tf.square(x0_pred - x0_true)) + tf.reduce_mean(tf.square(v0_pred - v0_true))