"""
train.py - PINNo Week1 trainer example (TensorFlow)

Place this file in PINNo/src/train.py

High-level classes:
- PDEProblem: sampling + analytic functions for your PDE.
- PINNModel: builds the neural network N(x,y) as a tf.keras.Model.
- Trainer: coordinates training, computes losses (PDE residual + BC),
           checkpointing, simple status reporting and prediction.

This code is intentionally simple and explicit to make it easy to read
and modify for a first run / demo.
"""

import os
import time
import threading
from typing import Tuple, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


# -------------------------
# PDE / problem definition
# -------------------------
class PDEProblem:
    """
    Encapsulates the PDE problem:
    - domain (here unit square [0,1]x[0,1])
    - analytical solution (for testing / BCs)
    - function A(x,y) used in trial solution
    - sampling routines for collocation (interior) and boundary points
    """

    def __init__(self, N: int = 100):
        """
        N: number of points per axis for full grid sampling (only used for evaluation/plotting)
        """
        self.N = N
        # domain limits
        self.xmin, self.xmax = 0.0, 1.0
        self.ymin, self.ymax = 0.0, 1.0

    # Analytical solution (numpy arrays accepted)
    @staticmethod
    def analytical_solution(x, y):
        """u(x,y) = exp(-x) * (x + y^3)"""
        # x,y might be numpy arrays or tf tensors; convert to numpy for analytic eval here
        return np.exp(-x) * (x + y ** 3)

    @staticmethod
    def A_func(x, y):
        """
        The A(x,y) function used to satisfy boundary conditions in the trial solution.
        Use same formula you provided (kept in numpy style).
        """
        # allow x,y to be numpy arrays
        return (
            (1 - x) * y ** 3
            + x * (1 + y ** 3) * np.exp(-1)
            + (1 - y) * x * (np.exp(-x) - np.exp(-1))
            + y * ((1 + x) * np.exp(-x) - (1 - x - 2 * x * np.exp(-1)))
        )

    def sample_interior(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample interior (collocation) points (x,y) in (xmin,xmax)x(ymin,ymax),
        excluding exact boundary points to avoid duplicating with BC samples.
        Returns x (n,1), y (n,1) as numpy arrays.
        """
        x = np.random.uniform(self.xmin, self.xmax, size=(n_points, 1))
        y = np.random.uniform(self.ymin, self.ymax, size=(n_points, 1))
        # Optionally remove points that lie exactly on boundary (floating point rare)
        return x, y

    def sample_boundary(self, n_per_edge: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample boundary points (Dirichlet): returns x (m,1), y (m,1), u_exact (m,1).
        We sample each edge uniformly with n_per_edge points.
        """
        xs = []
        ys = []

        # x = 0 edge
        ys0 = np.linspace(self.ymin, self.ymax, n_per_edge).reshape(-1, 1)
        xs0 = np.zeros_like(ys0)
        xs.append(xs0); ys.append(ys0)

        # x = 1 edge
        ys1 = ys0.copy()
        xs1 = np.ones_like(ys1)
        xs.append(xs1); ys.append(ys1)

        # y = 0 edge
        xs2 = np.linspace(self.xmin, self.xmax, n_per_edge).reshape(-1, 1)
        ys2 = np.zeros_like(xs2)
        xs.append(xs2); ys.append(ys2)

        # y = 1 edge
        xs3 = xs2.copy()
        ys3 = np.ones_like(xs3)
        xs.append(xs3); ys.append(ys3)

        x_b = np.vstack(xs)
        y_b = np.vstack(ys)
        u_b = self.analytical_solution(x_b, y_b).astype(np.float32)
        return x_b.astype(np.float32), y_b.astype(np.float32), u_b

    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return a structured grid for evaluation/plotting (N x N)."""
        x = np.linspace(self.xmin, self.xmax, self.N)
        y = np.linspace(self.ymin, self.ymax, self.N)
        X, Y = np.meshgrid(x, y)
        return X.reshape(-1, 1).astype(np.float32), Y.reshape(-1, 1).astype(np.float32)


# -------------------------
# PINN model (neural net)
# -------------------------
class PINNModel(tf.keras.Model):
    """
    Keras model that represents N(x,y) (the neural network inserted in the trial solution).
    We implement it as a tf.keras.Model so we can call it directly inside GradientTape.
    """

    def __init__(self, hidden_units: int = 40, n_hidden_layers: int = 2, activation="tanh"):
        super().__init__()
        self.hidden_layers = []
        act = tf.keras.activations.get(activation)
        # Input is 2 dims (x,y)
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(layers.Dense(hidden_units, activation=act))
        # Final output layer (1 value)
        self.out = layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        """
        Forward pass. inputs: tensor shape (batch, 2)
        """
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out(x)  # shape (batch, 1)


# -------------------------
# Trainer class
# -------------------------
class Trainer:
    """
    Trainer runs the custom PINN training loop.

    Public methods:
    - train(...) : run training (synchronous)
    - save_checkpoint(path)
    - load_checkpoint(path)
    - predict(x,y): compute trial solution on given points
    - get_status(): a small dict with current epoch, loss, running state
    """

    def __init__(
        self,
        problem: PDEProblem,
        model: PINNModel,
        learning_rate: float = 0.02,
        checkpoint_dir: str = "checkpoints",
    ):
        self.problem = problem
        self.model = model
        self.optimizer = optimizers.Adam(learning_rate)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # checkpoint objects
        self.ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=5)

        # training state
        self._running = False
        self._epoch = 0
        self._loss = np.inf
        self._stop_requested = False

    # -------------------------
    # Trial solution: Psi = A(x,y) + x(1-x)y(1-y)*N(x,y)
    # x,y are tf tensors with shape (batch,1)
    # -------------------------
    def trial_solution(self, x: tf.Tensor, y: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Build the trial solution as a tensor expression so derivatives can be computed.
        x,y: tf.Tensor shape (batch,1)
        returns Psi: tf.Tensor shape (batch,1)
        """
        # compute A(x,y) using tensorflow ops:
        # A expression originally given in numpy; we rewrite with tf ops.
        # For simplicity we compute A using numpy on CPU then convert to tf,
        # because A is algebraic and not requiring gradients (it is a fixed function).
        # But here we implement with tf to be consistent:
        # NOTE: if A is complicated and uses np-only functions, prefer computing it
        # in numpy and converting, but keep types float32.
        x_np = x.numpy() if isinstance(x, tf.Tensor) and not tf.executing_eagerly() else None

        # To keep everything TF-friendly, implement A using tf ops:
        # A(x,y) = (1-x)*y^3 + x*(1+y^3)*exp(-1) + (1-y)*x*(exp(-x)-exp(-1)) + y*((1+x)*exp(-x) - (1 - x - 2*x*exp(-1)))
        # Use tf.cast to ensure float32
        ex1 = tf.exp(-1.0)
        exp_minus_x = tf.exp(-x)
        Axy = (
            (1 - x) * tf.pow(y, 3)
            + x * (1 + tf.pow(y, 3)) * ex1
            + (1 - y) * x * (exp_minus_x - ex1)
            + y * ((1 + x) * exp_minus_x - (1 - x - 2 * x * ex1))
        )

        # Neural network N(x,y)
        xy = tf.concat([x, y], axis=1)  # shape (batch,2)
        N_xy = self.model(xy, training=training)  # shape (batch,1)

        Psi = Axy + x * (1 - x) * y * (1 - y) * N_xy
        return Psi

    # -------------------------
    # Loss computation: physics residual + boundary condition MSE
    # -------------------------
    def compute_loss(self, x_interior: tf.Tensor, y_interior: tf.Tensor, x_b: tf.Tensor, y_b: tf.Tensor, u_b: tf.Tensor) -> tf.Tensor:
        """
        Compute the total loss for a batch:
        - PDE residual MSE on interior collocation points
        - Dirichlet BC MSE on boundary points (trial solution should match analytical)
        """

        # Convert to float32 tensors
        x_int = tf.cast(x_interior, dtype=tf.float32)
        y_int = tf.cast(y_interior, dtype=tf.float32)
        x_b = tf.cast(x_b, dtype=tf.float32)
        y_b = tf.cast(y_b, dtype=tf.float32)
        u_b = tf.cast(u_b, dtype=tf.float32)

        # --- PDE residual using second derivatives via nested GradientTape ---
        # We'll use two tapes: inner for first derivatives, outer for second derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x_int, y_int])
            with tf.GradientTape() as tape1:
                tape1.watch([x_int, y_int])
                # compute trial solution (this calls the model inside tape so gradients flow)
                Psi = self.trial_solution(x_int, y_int, training=True)  # shape (batch,1)
            # first derivatives
            Psi_x = tape1.gradient(Psi, x_int)  # dPsi/dx
            Psi_y = tape1.gradient(Psi, y_int)  # dPsi/dy

        # second derivatives
        Psi_xx = tape2.gradient(Psi_x, x_int)
        Psi_yy = tape2.gradient(Psi_y, y_int)
        # free the persistent tape
        del tape2

        # const term on RHS of PDE: exp(-x) * (x - 2 + y^3 + 6*y)
        rhs = tf.exp(-x_int) * (x_int - 2.0 + tf.pow(y_int, 3) + 6.0 * y_int)

        # residual = Psi_xx + Psi_yy - rhs
        residual = Psi_xx + Psi_yy - rhs
        # PDE loss: mean squared residual
        pde_loss = tf.reduce_mean(tf.square(residual))

        # --- Boundary loss (Dirichlet) ---
        # compute trial solution on boundary points (no need for gradients here)
        Psi_b = self.trial_solution(x_b, y_b, training=False)
        bc_loss = tf.reduce_mean(tf.square(Psi_b - u_b))

        total_loss = pde_loss + bc_loss
        return total_loss, pde_loss, bc_loss

    # -------------------------
    # Single training step (gradient update)
    # -------------------------
    @tf.function
    def _train_step(self, x_int, y_int, x_b, y_b, u_b):
        """
        Note: this tf.function wraps the gradient calculation and apply_gradients.
        We separate the outer Python logic to allow logging and checkpointing at epoch boundaries.
        """
        with tf.GradientTape() as tape:
            total_loss, pde_loss, bc_loss = self.compute_loss(x_int, y_int, x_b, y_b, u_b)

        # collect trainable variables from the neural network model
        train_vars = self.model.trainable_variables
        grads = tape.gradient(total_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        return total_loss, pde_loss, bc_loss

    # -------------------------
    # Synchronous training loop
    # -------------------------
    def train(
        self,
        epochs: int = 1000,
        interior_batch_size: int = 10000,
        boundary_points_per_edge: int = 50,
        checkpoint_every: int = 100,
        verbose: bool = True,
    ):
        """
        Run training in the current thread (blocking).
        - interior_batch_size: how many collocation points per epoch
        - boundary_points_per_edge: how many points to sample on each boundary edge
        """
        self._running = True
        self._stop_requested = False

        # Pre-sample boundary points (you can re-sample each epoch if you want stochastic BC sampling)
        x_b_np, y_b_np, u_b_np = self.problem.sample_boundary(boundary_points_per_edge)
        x_b = tf.convert_to_tensor(x_b_np, dtype=tf.float32)
        y_b = tf.convert_to_tensor(y_b_np, dtype=tf.float32)
        u_b = tf.convert_to_tensor(u_b_np, dtype=tf.float32)

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            if self._stop_requested:
                if verbose:
                    print("Stop requested. Ending training early at epoch", epoch - 1)
                break

            # sample interior collocation points each epoch
            x_in_np, y_in_np = self.problem.sample_interior(interior_batch_size)
            x_in = tf.convert_to_tensor(x_in_np, dtype=tf.float32)
            y_in = tf.convert_to_tensor(y_in_np, dtype=tf.float32)

            # run a training step
            total_loss, pde_loss, bc_loss = self._train_step(x_in, y_in, x_b, y_b, u_b)

            # update state
            self._epoch = epoch
            self._loss = float(total_loss.numpy())

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch <= 5):
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch}/{epochs}] total_loss={self._loss:.6e}, pde_loss={float(pde_loss):.6e}, bc_loss={float(bc_loss):.6e}, elapsed={elapsed:.1f}s"
                )

            # checkpointing
            if epoch % checkpoint_every == 0:
                self.save_checkpoint()

        self._running = False
        if verbose:
            print("Training finished. Final epoch:", self._epoch, "final loss:", self._loss)

    def request_stop(self):
        """Tell the trainer to stop gracefully at the next epoch boundary."""
        self._stop_requested = True

    # -------------------------
    # Utilities: checkpointing, predict, status
    # -------------------------
    def save_checkpoint(self):
        """Save weights + optimizer state."""
        saved_path = self.manager.save()
        print("Saved checkpoint:", saved_path)
        return saved_path

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Load the latest or a specific checkpoint.
        If checkpoint_path is None, restore the latest managed checkpoint.
        """
        if checkpoint_path:
            self.ckpt.restore(checkpoint_path).expect_partial()
            print("Restored checkpoint:", checkpoint_path)
        else:
            latest = self.manager.latest_checkpoint
            if latest:
                self.ckpt.restore(latest).expect_partial()
                print("Restored latest checkpoint:", latest)
            else:
                print("No checkpoint found to restore.")

    def predict(self, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
        """
        Compute the trial solution Psi for the given numpy x,y arrays (shape (n,1)).
        Returns numpy array shape (n,1).
        """
        x_tf = tf.convert_to_tensor(x_np.astype(np.float32))
        y_tf = tf.convert_to_tensor(y_np.astype(np.float32))
        Psi_tf = self.trial_solution(x_tf, y_tf, training=False)
        return Psi_tf.numpy()

    def get_status(self) -> Dict:
        """Small status dict other code (e.g. GUI) can call."""
        return {"running": self._running, "epoch": int(self._epoch), "loss": float(self._loss)}


# -------------------------
# Example usage when run as script
# -------------------------
if __name__ == "__main__":
    # Quick demo - run a short training to validate everything works.
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs for demo")
    parser.add_argument("--hidden", type=int, default=40, help="Hidden units per layer")
    parser.add_argument("--batch", type=int, default=2000, help="Interior collocation points per epoch")
    args = parser.parse_args()

    # Create problem, model, trainer
    problem = PDEProblem(N=64)
    model = PINNModel(hidden_units=args.hidden, n_hidden_layers=2, activation="tanh")
    trainer = Trainer(problem, model, learning_rate=0.02, checkpoint_dir="checkpoints_demo")

    # Run training (synchronous)
    trainer.train(epochs=args.epochs, interior_batch_size=args.batch, boundary_points_per_edge=40, checkpoint_every=100)

    # After training, compute predictions on a grid and compare with analytic solution
    Xg, Yg = problem.grid()
    Psi_pred = trainer.predict(Xg, Yg).reshape(-1, 1)
    Psi_true = problem.analytical_solution(Xg, Yg).astype(np.float32)

    # Show a quick plot of the predicted solution surface (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        Nx = int(np.sqrt(Xg.shape[0]))
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(Xg.reshape(Nx, Nx), Yg.reshape(Nx, Nx), Psi_pred.reshape(Nx, Nx), cmap="viridis")
        ax1.set_title("Predicted Psi")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(Xg.reshape(Nx, Nx), Yg.reshape(Nx, Nx), Psi_true.reshape(Nx, Nx), cmap="viridis")
        ax2.set_title("Analytical Psi")

        plt.show()
    except Exception as e:
        print("Plotting failed:", e)
        print("You can inspect predictions by saving arrays or using other tools.")
