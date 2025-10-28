# gui_modules/training_tab.py
"""
Módulo para la pestaña de entrenamiento - Versión simplificada
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from src.config import get_active_config
from src.training import PINNTrainer
from gui_modules.components import TrainingVisualizer, MetricsCalculator


class TrainingTab(ttk.Frame):
    """Pestaña de entrenamiento simplificada"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.visualizer = TrainingVisualizer()
        self.metrics_calc = MetricsCalculator()
        
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
            values=["SHO", "DHO", "WAVE"], state="readonly"
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
        """Iniciar entrenamiento"""
        try:
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
        self.visualizer.init_plots()

    def stop_training(self):
        """Detener entrenamiento"""
        self.shared_state['is_training'] = False
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
        """Actualizar visualización del entrenamiento"""
        # Métricas en tiempo real
        if self.epoch % 10 == 0 or self.epoch == 1:
            self.epoch_label.config(text=f"Epoch: {self.epoch}")
            self.loss_label.config(text=f"Loss: {losses[0].numpy():.4e}")

        # Actualizar gráficas y métricas
        if self.epoch == 1 or self.epoch % 100 == 0:
            self._update_visualizations()

    def _update_visualizations(self):
        """Actualizar todas las visualizaciones"""
        trainer = self.shared_state['trainer']
        
        # Actualizar gráficas
        self.visualizer.update_loss_plot(self.epoch, self.loss_history)
        
        if hasattr(trainer, 'physics') and hasattr(trainer, 'model'):
            self._update_solution_visualization(trainer)
            self._update_metrics_summary(trainer)

    def _update_solution_visualization(self, trainer):
        """Actualizar visualización de solución"""
        problem = trainer.active_problem
        
        if problem in ("SHO", "DHO"):
            self._visualize_ode_solution(trainer)
        elif problem == "WAVE":
            self._visualize_pde_solution(trainer)

    def _visualize_ode_solution(self, trainer):
        """Visualizar solución de ODE"""
        t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
        t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
        
        x_pred = trainer.model(t_plot).numpy()
        x_true = trainer.physics.analytical_solution(t_plot)
        
        self.visualizer.update_solution_plot(t_plot, x_true, x_pred, "t", "x(t)")
        
        error = np.linalg.norm(x_pred - x_true) / (np.linalg.norm(x_true) + 1e-8)
        self.error_label.config(text=f"L2 Error: {error:.4f}")

    def _visualize_pde_solution(self, trainer):
        """Visualizar solución de PDE"""
        t_slice = 0.5
        x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
        x_plot = np.linspace(x_domain[0], x_domain[1], 100).reshape(-1, 1)
        
        xt_plot = np.hstack([x_plot, np.full_like(x_plot, t_slice)]).astype(np.float32)
        u_pred = trainer.model(xt_plot).numpy()
        u_true = trainer.physics.analytical_solution(xt_plot)
        
        self.visualizer.update_solution_plot(x_plot, u_true, u_pred, "x", f"u(x, t={t_slice})")
        
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
        else:  # WAVE
            x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
            x_plot = np.linspace(x_domain[0], x_domain[1], 100).reshape(-1, 1)
            xt_plot = np.hstack([x_plot, np.full_like(x_plot, 0.5)]).astype(np.float32)
            y_pred = trainer.model(xt_plot).numpy().flatten()
            y_true = trainer.physics.analytical_solution(xt_plot).flatten()
            x_data = x_plot

        # Generar reporte de métricas
        metrics_report = self.metrics_calc.comprehensive_report(y_true, y_pred)
        
        # Actualizar el panel de métricas en training tab
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, metrics_report)
        self.metrics_text.config(state="disabled")
    
        # ACTUALIZACIÓN NUEVA: Actualizar también el report tab
        self._update_report_tab(trainer, y_true, y_pred, x_data)

    def _handle_training_error(self, error):
        """Manejar errores de entrenamiento"""
        messagebox.showerror("Training Error", f"Error at epoch {self.epoch}:\n\n{error}")
        self.stop_training()

    def on_state_change(self, key, value):
        """Manejar cambios de estado global"""
        if key == 'is_training' and not value:
            self.stop_training()
    
    def _update_report_tab(self, trainer, y_true, y_pred, x_data):
        """Actualiza la pestaña de reportes con las métricas actuales"""
        # Calcular métricas para el reporte
        metrics_report = self.metrics_calc.comprehensive_report(y_true, y_pred)
    
        # Actualizar el report tab a través del estado compartido
        if hasattr(self, 'shared_state'):
            # Guardar el reporte en el estado compartido
            self.shared_state['last_metrics_report'] = metrics_report
            self.shared_state['last_plot_data'] = (x_data, y_true, y_pred)
            
            # Notificar al report tab
            if hasattr(self, 'report_tab_ref'):
                self.report_tab_ref.update_report(metrics_report, (x_data, y_true, y_pred))