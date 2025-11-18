"""
Módulo para la pestaña de entrenamiento
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.config import get_active_config
from src.training import PINNTrainer
from gui_modules.components import TrainingVisualizer, MetricsCalculator


class TrainingTab(ttk.Frame):
    """Pestaña de entrenamiento con conexión a ReportTab"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.visualizer = TrainingVisualizer()
        self.metrics_calc = MetricsCalculator()
        self.report_tab_ref = None  # Referencia a ReportTab
        
        self._build_interface()
        self._init_training_state()

    def _build_interface(self):
        """Construir interfaz de entrenamiento"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controles izquierdos
        self._build_control_panel(main_frame)
        
        # Visualización derecha
        self._build_visualization_panel(main_frame)

    def _build_control_panel(self, parent):
        """Panel de controles de entrenamiento"""
        control_frame = ttk.Labelframe(parent, text="Training Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        # Selector de problema
        ttk.Label(control_frame, text="Problem:").pack(pady=(0, 5), anchor="w")
        self.problem_selector = ttk.Combobox(
            control_frame, textvariable=self.shared_state['problem_name'],
            values=["SHO", "DHO", "HEAT"], state="readonly"
        )
        self.problem_selector.pack(pady=5, fill=tk.X)

        # Botones de control
        self.start_btn = ttk.Button(control_frame, text="Start Training", 
                                   command=self.start_training)
        self.start_btn.pack(pady=10, fill=tk.X)

        self.stop_btn = ttk.Button(control_frame, text="Stop Training",
                                  command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(pady=5, fill=tk.X)

        # Métricas en tiempo real
        self._build_live_metrics(control_frame)

    def _build_live_metrics(self, parent):
        """Mostrar métricas en tiempo real"""
        metrics_frame = ttk.Labelframe(parent, text="Live Metrics", padding="10")
        metrics_frame.pack(pady=20, fill=tk.X)
        
        self.epoch_label = ttk.Label(metrics_frame, text="Epoch: 0")
        self.epoch_label.pack(anchor="w")
        self.loss_label = ttk.Label(metrics_frame, text="Loss: N/A")
        self.loss_label.pack(anchor="w")
        self.error_label = ttk.Label(metrics_frame, text="L2 Error: N/A")
        self.error_label.pack(anchor="w")

    def _build_visualization_panel(self, parent):
        """Panel de visualización"""
        right_col = ttk.Frame(parent)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Gráficas de entrenamiento
        self.viz_frame = ttk.Frame(right_col)
        self.viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.visualizer.setup_plots(self.viz_frame)

        # Resumen de métricas
        self._build_metrics_summary(right_col)

    def _build_metrics_summary(self, parent):
        """Resumen detallado de métricas"""
        summary_frame = ttk.Labelframe(parent, text="Training Summary", padding=8)
        summary_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(8, 0))
        
        self.metrics_text = tk.Text(summary_frame, height=10, width=80)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Training summary will appear here...\n")
        self.metrics_text.config(state="disabled")

    def _init_training_state(self):
        """Inicializar estado del entrenamiento"""
        self.epoch = 0
        self.loss_history = []

    def start_training(self):
        """Iniciar entrenamiento - WITH MEMORY INIT"""
        try:
            # Reset plots only when starting new training
            self.visualizer.reset_plots()
            self.visualizer.init_plots()
            
            self._setup_trainer()
            self._update_ui_for_training_start()
            self._training_loop()
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to start training:\n\n{e}")
            self.stop_training()

    def _setup_trainer(self):
        """Configurar el entrenador PINN"""
        problem = self.shared_state['problem_name'].get()
        config = get_active_config(problem)
        self.shared_state['trainer'] = PINNTrainer(config, problem)
        self.shared_state['is_training'] = True

    def _update_ui_for_training_start(self):
        """Actualizar UI para inicio de entrenamiento"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.problem_selector.config(state=tk.DISABLED)
        self.epoch = 0
        self.loss_history = []

    def stop_training(self):
        """Detener entrenamiento - PRESERVE FINAL PLOT"""
        self.shared_state['is_training'] = False
        
        # DO NOT clear visualization - preserve final plot
        # Only clear TensorFlow session for memory
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.problem_selector.config(state=tk.NORMAL)

    def _training_loop(self):
        """Bucle principal de entrenamiento"""
        if not self.shared_state.get('is_training', False):
            return

        try:
            trainer = self.shared_state['trainer']
            losses = trainer.perform_one_step()
            
            # Actualizar estado
            self.epoch += 1
            total_loss = losses[0].numpy()
            self.loss_history.append(total_loss)
            
            # Actualizar UI
            self._update_training_display(losses)
            
            # Continuar bucle
            self.after(1, self._training_loop)
            
        except Exception as e:
            self._handle_training_error(e)

    def _update_training_display(self, losses):
        """Actualizar visualización del entrenamiento - OPTIMIZED"""
        # Lightweight updates every 10 epochs
        if self.epoch % 10 == 0 or self.epoch == 1:
            self.epoch_label.config(text=f"Epoch: {self.epoch}")
            self.loss_label.config(text=f"Loss: {losses[0].numpy():.4e}")

        # Medium updates (loss plot) every 100 epochs
        if self.epoch % 100 == 0:
            self.visualizer.update_loss_plot(self.epoch, self.loss_history)

        # HEAVY updates (solution plots) with optimized frequency
        current_problem = self.shared_state['problem_name'].get()
        if current_problem == "HEAT":
            update_solution = (self.epoch == 1 or self.epoch % 100 == 0)  # Reduced frequency
        else:
            # For ODE problems: update more frequently
            update_solution = (self.epoch == 1 or self.epoch % 100 == 0)
            
        if update_solution:
            self._update_visualizations()

    def _update_visualizations(self):
        """Actualizar todas las visualizaciones"""
        trainer = self.shared_state['trainer']
        
        # Update loss plot (lightweight)
        self.visualizer.update_loss_plot(self.epoch, self.loss_history)
        
        if hasattr(trainer, 'physics') and hasattr(trainer, 'model'):
            self._update_solution_visualization(trainer)
            self._update_metrics_summary(trainer)

    def _update_solution_visualization(self, trainer):
        """Actualizar visualización de solución"""
        problem = trainer.active_problem
        
        if problem in ("SHO", "DHO"):
            self._visualize_ode_solution(trainer)
        elif problem == "HEAT":
            self._visualize_heat_solution(trainer)

    def _visualize_ode_solution(self, trainer):
        """Visualizar solución de ODE"""
        t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
        t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
        
        x_pred = trainer.model(t_plot).numpy()
        x_true = trainer.physics.analytical_solution(t_plot)
        
        self.visualizer.update_solution_plot(t_plot, x_true, x_pred, "t", "x(t)")
        
        error = np.linalg.norm(x_pred - x_true) / (np.linalg.norm(x_true) + 1e-8)
        self.error_label.config(text=f"L2 Error: {error:.4f}")

    def _visualize_heat_solution(self, trainer):
        """Visualizar solución de calor 2D - OPTIMIZED WITH BETTER LAYOUT"""
        # Use consistent resolution for stable layout
        resolution = 20  # Fixed resolution for consistent performance
        
        t_slice = 0.5
        y_slice = 0.5
        x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
        t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
        
        # Create optimized grid with exact boundaries
        x_plot = np.linspace(x_domain[0], x_domain[1], resolution)
        t_plot = np.linspace(t_domain[0], t_domain[1], resolution)
        X, T = np.meshgrid(x_plot, t_plot)
        
        # Prepare points for prediction - ensure proper ordering
        xy_flat = np.stack([
            X.flatten(), 
            np.full_like(X.flatten(), y_slice), 
            T.flatten()
        ], axis=1).astype(np.float32)
        
        # Batch prediction for better performance
        u_pred = trainer.model(xy_flat).numpy().reshape(X.shape)
        
        # Use the fixed visualizer method with proper boundaries
        self.visualizer.update_heat_solution_plot(
            X, T, u_pred, 
            f"Heat Equation (y={y_slice}) - Epoch {self.epoch}"
        )
        
        # Calculate error for display
        u_true = trainer.physics.analytical_solution(xy_flat).reshape(X.shape)
        error = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
        self.error_label.config(text=f"L2 Error: {error:.4f}")

    def _update_metrics_summary(self, trainer):
        """Actualizar resumen de métricas"""
        problem = trainer.active_problem
        
        if problem in ("SHO", "DHO"):
            t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
            t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
            y_pred = trainer.model(t_plot).numpy().flatten()
            y_true = trainer.physics.analytical_solution(t_plot).flatten()
            x_data = t_plot
        elif problem == "HEAT":
            # Para calor, usamos un slice 2D en y fijo con lower resolution for metrics
            x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
            t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
            y_slice = 0.5
            
            # Use even lower resolution for metrics calculation
            x_plot = np.linspace(x_domain[0], x_domain[1], 20)
            t_plot = np.linspace(t_domain[0], t_domain[1], 20)
            X, T = np.meshgrid(x_plot, t_plot)
            xy_flat = np.stack([X.flatten(), np.full_like(X.flatten(), y_slice), T.flatten()], axis=1)
            xy_flat = xy_flat.astype(np.float32)
            
            y_pred = trainer.model(xy_flat).numpy().flatten()
            y_true = trainer.physics.analytical_solution(xy_flat).flatten()
            x_data = X.flatten()

        # Generar reporte de métricas
        metrics_report = self.metrics_calc.comprehensive_report(y_true, y_pred)
        
        # Actualizar el panel de métricas en training tab
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, metrics_report)
        self.metrics_text.config(state="disabled")
        
        # Actualizar también el report tab
        self._update_report_tab(trainer, y_true, y_pred, x_data)

    def _update_report_tab(self, trainer, y_true, y_pred, x_data):
        """Actualiza la pestaña de reportes con las métricas actuales"""
        # Calcular métricas para el reporte
        metrics_report = self.metrics_calc.comprehensive_report(y_true, y_pred)
        
        # Actualizar el report tab si existe la referencia
        if self.report_tab_ref is not None:
            try:
                self.report_tab_ref.update_report(metrics_report, (x_data, y_true, y_pred))
            except Exception as e:
                print(f"Error updating report tab: {e}")

    def _handle_training_error(self, error):
        """Manejar errores de entrenamiento"""
        messagebox.showerror("Training Error", f"Error at epoch {self.epoch}:\n\n{error}")
        self.stop_training()

    def on_shared_state_change(self, key, value):
        """Manejar cambios de estado global"""
        if key == 'is_training' and not value:
            self.stop_training()