# gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

from src.config import get_active_config
from src.training import PINNTrainer

class PINNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PINN Interactive Trainer")
        self.root.geometry("1200x700")

        self.trainer = None
        self.is_training = False
        self.epoch = 0
        self.loss_history = []
        self.problem_name = tk.StringVar(value="SHO")

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        control_frame = ttk.Labelframe(main_frame, text="Controles", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(control_frame, text="Seleccionar Problema:").pack(pady=(0, 5), anchor="w")
        self.problem_selector = ttk.Combobox(control_frame, textvariable=self.problem_name, values=["SHO", "DHO", "WAVE"], state="readonly")
        self.problem_selector.pack(pady=5, fill=tk.X)

        self.start_button = ttk.Button(control_frame, text="Iniciar Entrenamiento", command=self.start_training)
        self.start_button.pack(pady=10, fill=tk.X)
        self.stop_button = ttk.Button(control_frame, text="Detener Entrenamiento", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(pady=5, fill=tk.X)

        metrics_frame = ttk.Labelframe(control_frame, text="Métricas", padding="10")
        metrics_frame.pack(pady=20, fill=tk.X)
        self.epoch_label = ttk.Label(metrics_frame, text="Época: 0")
        self.epoch_label.pack(anchor="w")
        self.loss_label = ttk.Label(metrics_frame, text="Pérdida (Loss): N/A")
        self.loss_label.pack(anchor="w")
        self.error_label = ttk.Label(metrics_frame, text="Error Relativo L2: N/A")
        self.error_label.pack(anchor="w")

        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_loss = self.fig.add_subplot(2, 1, 1)
        self.ax_solution = self.fig.add_subplot(2, 1, 2)
        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.init_plots()

    def init_plots(self):
        self.ax_loss.clear()
        self.ax_loss.set_title("Función de Pérdida vs. Épocas")
        self.ax_loss.set_xlabel("Época")
        self.ax_loss.set_ylabel("Pérdida (escala log)")
        self.ax_loss.grid(True, which="both", linestyle='--', linewidth=0.5)
        self.loss_line, = self.ax_loss.plot([], [], 'b-')
        self.ax_solution.clear()
        self.ax_solution.set_title("Solución Predicha vs. Analítica")
        self.ax_solution.grid(True, linestyle='--', linewidth=0.5)
        self.pred_line, = self.ax_solution.plot([], [], 'b-', label="Predicción PINN")
        self.true_line, = self.ax_solution.plot([], [], 'r--', label="Solución Analítica")
        self.ax_solution.legend()
        self.canvas.draw()

    def start_training(self):
        self.epoch = 0
        self.loss_history = []
        self.is_training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.problem_selector.config(state=tk.DISABLED)
        
        try:
            problem = self.problem_name.get()
            print(f"--- Cargando configuración para: {problem} ---")
            active_config = get_active_config(problem)
            print("--- Inicializando PINNTrainer ---")
            self.trainer = PINNTrainer(active_config, problem) # <-- LLAMADA CORRECTA
            print("--- Trainer inicializado correctamente ---")
        except Exception as e:
            messagebox.showerror("Error de Inicialización", f"No se pudo iniciar el entrenador:\n\n{e}")
            self.stop_training()
            return

        self.init_plots()
        print(f"--- Iniciando entrenamiento para: {problem} ---")
        self.training_loop()

    def stop_training(self):
        self.is_training = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.problem_selector.config(state=tk.NORMAL)
        print("--- Entrenamiento detenido por el usuario ---")

    def training_loop(self):
        if not self.is_training: return
        try:
            losses = self.trainer.perform_one_step()
            total_loss = losses[0].numpy()
            self.loss_history.append(total_loss)
            self.epoch += 1
            if self.epoch % 10 == 0 or self.epoch == 1:
                self.epoch_label.config(text=f"Época: {self.epoch}")
                self.loss_label.config(text=f"Pérdida (Loss): {total_loss:.4e}")
            if self.epoch % 100 == 0 or self.epoch == 1:
                self.update_plots()
            self.root.after(1, self.training_loop)
        except Exception as e:
            print(f"ERROR en el bucle de entrenamiento en la época {self.epoch}: {e}")
            messagebox.showerror("Error en Entrenamiento", f"Ocurrió un error en la época {self.epoch}:\n\n{e}")
            self.stop_training()

    def update_plots(self):
        epochs_data = range(len(self.loss_history))
        self.loss_line.set_data(epochs_data, self.loss_history)
        self.ax_loss.set_yscale('log')
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        problem = self.trainer.active_problem
        if problem in ("SHO", "DHO"):
            t_domain = self.trainer.config["PHYSICS_CONFIG"]["t_domain"]
            t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
            x_pred = self.trainer.model(t_plot).numpy()
            x_true = self.trainer.physics.analytical_solution(t_plot)
            self.pred_line.set_data(t_plot, x_pred)
            self.true_line.set_data(t_plot, x_true)
            self.ax_solution.set_xlabel("t")
            self.ax_solution.set_ylabel("x(t)")
            error = np.linalg.norm(x_pred - x_true) / (np.linalg.norm(x_true) + 1e-8)
            self.error_label.config(text=f"Error Relativo L2: {error:.4f}")
        elif problem == "WAVE":
            t_slice = 0.5
            x_domain = self.trainer.config["PHYSICS_CONFIG"]["x_domain"]
            x_plot = np.linspace(x_domain[0], x_domain[1], 100).reshape(-1, 1)
            xt_plot = np.hstack([x_plot, np.full_like(x_plot, t_slice)]).astype(np.float32)
            u_pred = self.trainer.model(xt_plot).numpy()
            u_true = self.trainer.physics.analytical_solution(xt_plot)
            self.pred_line.set_data(x_plot, u_pred)
            self.true_line.set_data(x_plot, u_true)
            self.ax_solution.set_title(f"Solución para t={t_slice}")
            self.ax_solution.set_xlabel("x")
            self.ax_solution.set_ylabel(f"u(x, t={t_slice})")
            error = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            self.error_label.config(text=f"Error Relativo L2: {error:.4f}")
        self.ax_solution.relim()
        self.ax_solution.autoscale_view()
        self.canvas.draw()

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    root = tk.Tk()
    app = PINNGUI(root)
    root.mainloop()