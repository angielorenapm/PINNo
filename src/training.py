# src/training.py
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List
import numpy as np
import tensorflow as tf

from src.models import get_model
from src.physics import get_physics_problem

# (Aquí van todas tus funciones de sampleo y de pérdidas, cópialas de tu archivo original)
# ... sample_collocation_sho, initial_loss_sho, etc. ...
def sample_collocation_sho(t_domain: Tuple[float, float], n_collocation: int) -> np.ndarray:
    t_min, t_max = t_domain
    t = np.random.uniform(t_min, t_max, size=(n_collocation, 1)).astype(np.float32)
    return t
def sample_initial_sho(t0: float = 0.0, n_initial: int = 1) -> np.ndarray:
    t0_arr = np.full((n_initial, 1), fill_value=float(t0), dtype=np.float32)
    return t0_arr
def sample_collocation_wave(x_domain: Tuple[float, float], t_domain: Tuple[float, float], n_collocation: int) -> np.ndarray:
    x = np.random.uniform(x_domain[0], x_domain[1], size=(n_collocation, 1))
    t = np.random.uniform(t_domain[0], t_domain[1], size=(n_collocation, 1))
    xt = np.hstack([x, t]).astype(np.float32)
    return xt
def sample_initial_wave(x_domain: Tuple[float, float], n_initial: int) -> np.ndarray:
    x = np.random.uniform(x_domain[0], x_domain[1], size=(n_initial, 1))
    t0 = np.zeros_like(x)
    xt0 = np.hstack([x, t0]).astype(np.float32)
    return xt0
def sample_boundary_wave(x_values: Tuple[float, float], t_domain: Tuple[float, float], n_boundary: int) -> np.ndarray:
    n_half = n_boundary // 2
    t1 = np.random.uniform(t_domain[0], t_domain[1], size=(n_half, 1))
    t2 = np.random.uniform(t_domain[0], t_domain[1], size=(n_boundary - n_half, 1))
    x_min_arr = np.full_like(t1, fill_value=x_values[0])
    x_max_arr = np.full_like(t2, fill_value=x_values[1])
    xt_b1 = np.hstack([x_min_arr, t1])
    xt_b2 = np.hstack([x_max_arr, t2])
    xt_b = np.vstack([xt_b1, xt_b2]).astype(np.float32)
    return xt_b
def initial_loss_sho(model: tf.keras.Model, t0: tf.Tensor, x0_true: float, v0_true: float) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(t0)
        x0_pred = model(t0)
    v0_pred = tape.gradient(x0_pred, t0)
    loss_x0 = tf.reduce_mean(tf.square(x0_pred - x0_true))
    loss_v0 = tf.reduce_mean(tf.square(v0_pred - v0_true))
    return loss_x0 + loss_v0
def initial_loss_wave(model: tf.keras.Model, xt0: tf.Tensor) -> tf.Tensor:
    x = xt0[:, 0:1]
    u_pred = model(xt0)
    u_true = tf.sin(np.pi * x)
    loss_u = tf.reduce_mean(tf.square(u_pred - u_true))
    xt0_var = tf.convert_to_tensor(xt0)
    with tf.GradientTape() as tape:
        tape.watch(xt0_var)
        u = model(xt0_var)
    u_t = tape.gradient(u, xt0_var)[:, 1:2]
    loss_ut = tf.reduce_mean(tf.square(u_t))
    return loss_u + loss_ut
def boundary_loss_wave(model: tf.keras.Model, xt_b: tf.Tensor) -> tf.Tensor:
    u_b_pred = model(xt_b)
    loss_b = tf.reduce_mean(tf.square(u_b_pred))
    return loss_b

class PINNTrainer:
    def __init__(self, config_dict: Dict[str, Any], problem_name: str):
        self.config = config_dict
        self.active_problem = problem_name

        model_config = dict(self.config["MODEL_CONFIG"])
        self.model = get_model(self.config["MODEL_NAME"], model_config)
        self.physics = get_physics_problem(self.active_problem, {"PHYSICS_CONFIG": self.config["PHYSICS_CONFIG"]})
        self.optimizer = tf.keras.optimizers.Adam(self.config["LEARNING_RATE"])
        self.loss_weights = self.config["LOSS_WEIGHTS"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config["RESULTS_PATH"], f"{self.config['RUN_NAME']}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self._prepare_data()

    def _prepare_data(self):
        phys = self.config["PHYSICS_CONFIG"]
        data_cfg = self.config["DATA_CONFIG"]

        if self.active_problem in ("SHO", "DHO"):
            t_domain = phys["t_domain"]
            n_coll = int(data_cfg["n_collocation"])
            self.t_coll = tf.convert_to_tensor(sample_collocation_sho(t_domain, n_coll), dtype=tf.float32)
            self.t0 = tf.convert_to_tensor(sample_initial_sho(t_domain[0], 1), dtype=tf.float32)
            self.x0_true = tf.constant(phys["initial_conditions"]["x0"], dtype=tf.float32)
            self.v0_true = tf.constant(phys["initial_conditions"]["v0"], dtype=tf.float32)
        elif self.active_problem == "WAVE":
            x_domain = phys["x_domain"]
            t_domain = phys["t_domain"]
            n_coll, n_init, n_bnd = data_cfg["n_collocation"], data_cfg["n_initial"], data_cfg["n_boundary"]
            self.xt_coll = tf.convert_to_tensor(sample_collocation_wave(x_domain, t_domain, n_coll), dtype=tf.float32)
            self.xt0 = tf.convert_to_tensor(sample_initial_wave(x_domain, n_init), dtype=tf.float32)
            self.xt_b = tf.convert_to_tensor(sample_boundary_wave(x_domain, t_domain, n_bnd), dtype=tf.float32)

    @tf.function
    def train_step_sho(self, t_coll, t0, x0_true, v0_true):
        with tf.GradientTape() as tape:
            residual = self.physics.pde_residual(self.model, t_coll)
            loss_pde = tf.reduce_mean(tf.square(residual))
            loss_init = initial_loss_sho(self.model, t0, x0_true, v0_true)
            loss = self.loss_weights["ode"] * loss_pde + self.loss_weights["initial"] * loss_init
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, loss_pde, loss_init

    @tf.function
    def train_step_wave(self, xt_coll, xt0, xt_b):
        with tf.GradientTape() as tape:
            loss_pde = tf.reduce_mean(tf.square(self.physics.pde_residual(self.model, xt_coll)))
            loss_init = initial_loss_wave(self.model, xt0)
            loss_bnd = boundary_loss_wave(self.model, xt_b)
            loss = self.loss_weights["pde"] * loss_pde + self.loss_weights["initial"] * loss_init + self.loss_weights["boundary"] * loss_bnd
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, loss_pde, loss_init, loss_bnd

    def perform_one_step(self) -> List[tf.Tensor]:
        if self.active_problem in ("SHO", "DHO"):
            return self.train_step_sho(self.t_coll, self.t0, self.x0_true, self.v0_true)
        elif self.active_problem == "WAVE":
            return self.train_step_wave(self.xt_coll, self.xt0, self.xt_b)