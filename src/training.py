# src/training.py
"""
Entrenador para PINNs (Physics-Informed Neural Networks).

Este archivo provee un Trainer simple pero funcional que:
- carga la configuración desde src.config
- instancia el modelo desde src.models
- instancia la física desde src.physics (solo para pde_residual)
- construye datasets (puntos iniciales, collocation y frontera)
- entrena la red minimizando la combinación de pérdidas:
    loss = w_pde * MSE(residual) + w_init * MSE(initial) + w_bnd * MSE(boundary)

Notas:
- Evito depender de initial_loss / boundary_loss de las clases de physics,
  porque en el código provisto hay inconsistencias (atributos con distinto nombre
  y falta de import np en physics.py). En su lugar calculo esas pérdidas aquí.
- Si más adelante arreglas physics.py, se puede delegar de nuevo a esos métodos.
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf

from src import config
from src.models import get_model
from src.physics import get_physics_problem

# -----------------------
# Utilidades para datos
# -----------------------
def sample_collocation_sho(t_domain: Tuple[float, float], n_collocation: int) -> np.ndarray:
    t_min, t_max = t_domain
    t = np.random.uniform(t_min, t_max, size=(n_collocation, 1)).astype(np.float32)
    return t

def sample_initial_sho(t0: float = 0.0, n_initial: int = 1) -> np.ndarray:
    # normalmente n_initial == 1 for SHO/DHO
    t0_arr = np.full((n_initial, 1), fill_value=float(t0), dtype=np.float32)
    return t0_arr

def sample_collocation_wave(x_domain: Tuple[float, float], t_domain: Tuple[float, float], n_collocation: int) -> np.ndarray:
    x = np.random.uniform(x_domain[0], x_domain[1], size=(n_collocation, 1))
    t = np.random.uniform(t_domain[0], t_domain[1], size=(n_collocation, 1))
    xt = np.hstack([x, t]).astype(np.float32)
    return xt

def sample_initial_wave(x_domain: Tuple[float, float], n_initial: int) -> np.ndarray:
    # initial at t = 0
    x = np.random.uniform(x_domain[0], x_domain[1], size=(n_initial, 1))
    t0 = np.zeros_like(x)
    xt0 = np.hstack([x, t0]).astype(np.float32)
    return xt0

def sample_boundary_wave(x_values: Tuple[float, float], t_domain: Tuple[float, float], n_boundary: int) -> np.ndarray:
    # sample half at x=x_min and half at x=x_max
    n_half = n_boundary // 2
    t1 = np.random.uniform(t_domain[0], t_domain[1], size=(n_half, 1))
    t2 = np.random.uniform(t_domain[0], t_domain[1], size=(n_boundary - n_half, 1))
    x_min_arr = np.full_like(t1, fill_value=x_values[0])
    x_max_arr = np.full_like(t2, fill_value=x_values[1])
    xt_b1 = np.hstack([x_min_arr, t1])
    xt_b2 = np.hstack([x_max_arr, t2])
    xt_b = np.vstack([xt_b1, xt_b2]).astype(np.float32)
    return xt_b

# -----------------------
# Pérdidas específicas
# -----------------------
def initial_loss_sho(model: tf.keras.Model, t0: tf.Tensor, x0_true: float, v0_true: float) -> tf.Tensor:
    """
    Calcula L_initial = MSE(x(t0) - x0_true) + MSE(x_t(t0) - v0_true)
    """
    with tf.GradientTape() as tape:
        tape.watch(t0)
        x0_pred = model(t0)  # shape (N,1)
    v0_pred = tape.gradient(x0_pred, t0)
    loss_x0 = tf.reduce_mean(tf.square(x0_pred - x0_true))
    loss_v0 = tf.reduce_mean(tf.square(v0_pred - v0_true))
    return loss_x0 + loss_v0

def initial_loss_wave(model: tf.keras.Model, xt0: tf.Tensor) -> tf.Tensor:
    """
    Para el caso de la prueba: u(x,0) = sin(pi x), u_t(x,0) = 0
    xt0: (N,2) con la segunda columna = 0
    """
    x = xt0[:, 0:1]
    # u_pred
    u_pred = model(xt0)
    u_true = tf.sin(np.pi * x)
    loss_u = tf.reduce_mean(tf.square(u_pred - u_true))

    # u_t pred: necesitamos cintar grad wrt t
    # Para ello, re-creamos un tensor de entrada y watch las componentes de t
    xt0_var = tf.convert_to_tensor(xt0)
    with tf.GradientTape() as tape:
        tape.watch(xt0_var)
        u = model(xt0_var)
    # derivada w.r.t t está en columna 1
    u_t = tape.gradient(u, xt0_var)[:, 1:2]
    loss_ut = tf.reduce_mean(tf.square(u_t))
    return loss_u + loss_ut

def boundary_loss_wave(model: tf.keras.Model, xt_b: tf.Tensor) -> tf.Tensor:
    """
    Para condiciones de frontera homogéneas u(0,t)=0 y u(1,t)=0
    xt_b shape (N,2)
    """
    u_b_pred = model(xt_b)
    loss_b = tf.reduce_mean(tf.square(u_b_pred))
    return loss_b

# -----------------------
# Trainer
# -----------------------
class PINNTrainer:
    def __init__(self, cfg_module):
        # Config module (src.config)
        self.cfg = cfg_module

        # active problem string (ej. "SHO")
        self.active_problem = getattr(self.cfg, "ACTIVE_PROBLEM", None)
        if self.active_problem is None:
            raise ValueError("ACTIVE_PROBLEM no definido en config.py")

        # Model
        model_config = dict(self.cfg.MODEL_CONFIG)  # copia
        self.model = get_model(self.cfg.MODEL_NAME, model_config)

        # Physics problem (usaremos la clase para pde_residual)
        # get_physics_problem espera un dict con key "PHYSICS_CONFIG"
        self.physics = get_physics_problem(self.active_problem, {"PHYSICS_CONFIG": self.cfg.PHYSICS_CONFIG})

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.cfg.LEARNING_RATE)

        # Loss weights
        self.loss_weights = self.cfg.LOSS_WEIGHTS

        # Results path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.cfg.RESULTS_PATH, f"{self.cfg.RUN_NAME}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Training parameters
        self.epochs = int(self.cfg.EPOCHS)

        # Prepare data (sample once; could resample each epoch if desired)
        self._prepare_data()

        # Checkpoint manager
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, directory=self.run_dir, max_to_keep=3)
        # try restore
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"[Trainer] Checkpoint restaurado: {self.ckpt_manager.latest_checkpoint}")

    def _prepare_data(self):
        """Construye los tensores de puntos (collocation, initial, boundary) según el problema activo."""
        phys = self.cfg.PHYSICS_CONFIG
        data_cfg = self.cfg.DATA_CONFIG

        if self.active_problem in ("SHO", "DHO"):
            t_domain = phys["t_domain"]
            n_coll = int(data_cfg["n_collocation"])
            n_init = int(data_cfg.get("n_initial", 1))
            self.t_coll = tf.convert_to_tensor(sample_collocation_sho(t_domain, n_coll), dtype=tf.float32)
            self.t0 = tf.convert_to_tensor(sample_initial_sho(t_domain[0], n_init), dtype=tf.float32)

            # true ICs
            self.x0_true = tf.constant(phys["initial_conditions"]["x0"], dtype=tf.float32)
            self.v0_true = tf.constant(phys["initial_conditions"]["v0"], dtype=tf.float32)

        elif self.active_problem == "WAVE":
            x_domain = phys["x_domain"]
            t_domain = phys["t_domain"]
            n_coll = int(data_cfg["n_collocation"])
            n_init = int(data_cfg["n_initial"])
            n_bnd = int(data_cfg["n_boundary"])

            self.xt_coll = tf.convert_to_tensor(sample_collocation_wave(x_domain, t_domain, n_coll), dtype=tf.float32)
            self.xt0 = tf.convert_to_tensor(sample_initial_wave(x_domain, n_init), dtype=tf.float32)
            self.xt_b = tf.convert_to_tensor(sample_boundary_wave(x_domain, t_domain, n_bnd), dtype=tf.float32)
        else:
            raise ValueError(f"Problema {self.active_problem} no soportado en _prepare_data().")

    @tf.function
    def train_step_sho(self, t_coll, t0, x0_true, v0_true):
        """
        paso de entrenamiento para SHO/DHO
        """
        with tf.GradientTape() as tape:
            # pde residual
            residual = self.physics.pde_residual(self.model, t_coll)  # espera (N,1)
            loss_pde = tf.reduce_mean(tf.square(residual))

            # initial loss (calculado localmente)
            loss_init = initial_loss_sho(self.model, t0, x0_true, v0_true)

            # total
            w_ode = tf.cast(self.loss_weights.get("ode", 1.0), tf.float32)
            w_init = tf.cast(self.loss_weights.get("initial", 1.0), tf.float32)
            loss = w_ode * loss_pde + w_init * loss_init

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, loss_pde, loss_init

    @tf.function
    def train_step_wave(self, xt_coll, xt0, xt_b):
        with tf.GradientTape() as tape:
            residual = self.physics.pde_residual(self.model, xt_coll)  # expects (N,2) input
            loss_pde = tf.reduce_mean(tf.square(residual))

            loss_init = initial_loss_wave(self.model, xt0)
            loss_bnd = boundary_loss_wave(self.model, xt_b)

            w_pde = tf.cast(self.loss_weights.get("pde", 1.0), tf.float32)
            w_init = tf.cast(self.loss_weights.get("initial", 1.0), tf.float32)
            w_bnd = tf.cast(self.loss_weights.get("boundary", 1.0), tf.float32)

            loss = w_pde * loss_pde + w_init * loss_init + w_bnd * loss_bnd

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, loss_pde, loss_init, loss_bnd

    def train(self):
        print(f"[Trainer] Iniciando entrenamiento: run_dir={self.run_dir}")
        start_time = time.time()

        best_loss = np.inf
        for epoch in range(1, self.epochs + 1):
            if self.active_problem in ("SHO", "DHO"):
                loss, loss_pde, loss_init = self.train_step_sho(self.t_coll, self.t0, self.x0_true, self.v0_true)
                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch {epoch:6d} | loss={loss.numpy():.6e} pde={loss_pde.numpy():.6e} init={loss_init.numpy():.6e}")

            elif self.active_problem == "WAVE":
                loss, loss_pde, loss_init, loss_bnd = self.train_step_wave(self.xt_coll, self.xt0, self.xt_b)
                if epoch % 100 == 0 or epoch == 1:
                    print(f"Epoch {epoch:6d} | loss={loss.numpy():.6e} pde={loss_pde.numpy():.6e} init={loss_init.numpy():.6e} bnd={loss_bnd.numpy():.6e}")

            # checkpoint cada cierto número de épocas
            if epoch % 2000 == 0:
                saved_path = self.ckpt_manager.save()
                print(f"[Trainer] Checkpoint guardado en {saved_path}")

            # guarda el mejor modelo
            if loss.numpy() < best_loss:
                best_loss = float(loss.numpy())
                # guardo pesos en formato Keras
                weights_path = os.path.join(self.run_dir, "best_weights.weights.h5")
                self.model.save_weights(weights_path)
                # simple log
                # (guardar con frecuencia alta puede ocupar espacio; lo hago solo cuando mejora)
        elapsed = time.time() - start_time
        print(f"[Trainer] Entrenamiento finalizado en {elapsed:.1f}s. Mejor loss={best_loss:.6e}")
        final_weights = os.path.join(self.run_dir, "final_weights.weights.h5")
        self.model.save_weights(final_weights)
        print(f"[Trainer] Pesos finales guardados en: {final_weights}")

# -----------------------
# Runner (script)
# -----------------------
def main():
    # opcional: semilla
    np.random.seed(1234)
    tf.random.set_seed(1234)

    trainer = PINNTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
