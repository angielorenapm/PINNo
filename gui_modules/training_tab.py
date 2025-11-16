"""
Módulo para la pestaña de entrenamiento - PROPER DIMENSIONS WITH PLOTLY
(Versión 0.0.4 - Con diálogo de selección de columnas)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import io

from src.config import get_active_config, get_problem_variables
from src.training import PINNTrainer
from gui_modules.components import TrainingVisualizer, MetricsCalculator


class TrainingTab(ttk.Frame):
    """Pestaña de entrenamiento con diálogo de selección de columnas"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.visualizer = TrainingVisualizer()
        self.metrics_calc = MetricsCalculator()
        self.report_tab_ref = None
        
        self.plot_label = None
        self.column_mapping = {}  # Almacena los mapeos por problema
        
        self._build_interface()
        self._init_training_state()

    def _build_interface(self):
        """Construir interfaz de entrenamiento"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._build_control_panel(main_frame)
        self._build_visualization_panel(main_frame)

    def _build_control_panel(self, parent):
        """Panel de controles de entrenamiento"""
        control_frame = ttk.Labelframe(parent, text="Training Controls", padding="8")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Selector de problema
        ttk.Label(control_frame, text="Problem:").pack(pady=(0, 3), anchor="w")
        self.problem_selector = ttk.Combobox(
            control_frame, textvariable=self.shared_state['problem_name'],
            values=["SHO", "DHO", "HEAT"], state="readonly", width=12
        )
        self.problem_selector.pack(pady=3, fill=tk.X)
        self.problem_selector.bind('<<ComboboxSelected>>', self._on_problem_change)

        # Botón para mapear columnas
        self.map_btn = ttk.Button(control_frame, text="Map CSV Columns...", 
                                 command=self._prompt_column_mapping)
        self.map_btn.pack(pady=5, fill=tk.X)

        # Indicador de modo
        self.mode_label = ttk.Label(control_frame, text="Mode: Analytical", 
                                   font=('Arial', 9, 'bold'), foreground='blue')
        self.mode_label.pack(pady=5)

        # Información de mapeo
        self.mapping_info = ttk.Label(control_frame, text="No CSV mapping", 
                                     font=('Arial', 8), foreground='gray', wraplength=200)
        self.mapping_info.pack(pady=5)

        # Botones de control
        self.start_btn = ttk.Button(control_frame, text="Start Training", 
                                   command=self.start_training)
        self.start_btn.pack(pady=8, fill=tk.X)

        self.stop_btn = ttk.Button(control_frame, text="Stop Training",
                                  command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(pady=3, fill=tk.X)

        # Métricas en tiempo real
        self._build_live_metrics(control_frame)

    def _build_live_metrics(self, parent):
        """Mostrar métricas en tiempo real"""
        metrics_frame = ttk.Labelframe(parent, text="Live Metrics", padding="8")
        metrics_frame.pack(pady=15, fill=tk.X)
        
        self.epoch_label = ttk.Label(metrics_frame, text="Epoch: 0", font=('Arial', 9))
        self.epoch_label.pack(anchor="w")
        self.loss_label = ttk.Label(metrics_frame, text="Loss: N/A", font=('Arial', 9))
        self.loss_label.pack(anchor="w")
        self.error_label = ttk.Label(metrics_frame, text="Error: N/A", font=('Arial', 9))
        self.error_label.pack(anchor="w")

    def _build_visualization_panel(self, parent):
        """Panel de visualización con Plotly"""
        right_col = ttk.Frame(parent)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.visualizer.setup_plots()
        
        self.plot_label = ttk.Label(right_col)
        self.plot_label.pack(fill=tk.BOTH, expand=True)
        
        self._update_plot_display()

        self._build_metrics_summary(right_col)

    def _build_metrics_summary(self, parent):
        """Resumen detallado de métricas"""
        summary_frame = ttk.Labelframe(parent, text="Training Summary", padding=6)
        summary_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(6, 0))
        
        self.metrics_text = tk.Text(summary_frame, height=8, width=60, font=('Arial', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Training summary will appear here...\n")
        self.metrics_text.config(state="disabled")

    def _update_plot_display(self):
        """Update the Tkinter label with the current Plotly image"""
        try:
            pil_image = self.visualizer.get_plot_image()
            photo = ImageTk.PhotoImage(pil_image)
            self.plot_label.configure(image=photo)
            self.plot_label.image = photo
        except Exception as e:
            print(f"Error updating plot display: {e}")

    def _init_training_state(self):
        """Inicializar estado del entrenamiento"""
        self.epoch = 0
        self.loss_history = []

    def _on_problem_change(self, event=None):
        """Handle problem selection change"""
        self._update_mode_display()

    def _update_mode_display(self):
        """Update the mode display based on current data and mapping"""
        problem = self.shared_state['problem_name'].get()
        csv_data = self.shared_state.get('current_dataframe')
        
        if csv_data is not None and problem in self.column_mapping:
            mapping = self.column_mapping[problem]
            required_vars = get_problem_variables(problem)
            
            if all(var in mapping for var in required_vars):
                self.mode_label.config(text="Mode: CSV Data", foreground='green')
                mapping_text = ", ".join([f"{k}: {v}" for k, v in mapping.items()])
                self.mapping_info.config(text=f"Mapping: {mapping_text}")
            else:
                self.mode_label.config(text="Mode: Analytical (incomplete mapping)", foreground='orange')
                self.mapping_info.config(text="Click 'Map CSV Columns' to configure")
        else:
            self.mode_label.config(text="Mode: Analytical", foreground='blue')
            if csv_data is not None:
                self.mapping_info.config(text="Click 'Map CSV Columns' to use CSV data")
            else:
                self.mapping_info.config(text="No CSV data loaded")

    def _prompt_column_mapping(self):
        """Diálogo para que el usuario mapee las columnas del CSV"""
        problem = self.shared_state['problem_name'].get()
        csv_data = self.shared_state.get('current_dataframe')
        
        if csv_data is None:
            messagebox.showinfo("No CSV Data", "Please load a CSV file first in the Data Exploration tab.")
            return
        
        required_vars = get_problem_variables(problem)
        available_columns = list(csv_data.columns)
        
        # Crear diálogo
        dialog = tk.Toplevel(self)
        dialog.title(f"Map Columns for {problem}")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Map CSV columns to {problem} variables:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Frame para selecciones
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        selections = {}
        
        for i, var_name in enumerate(required_vars):
            ttk.Label(selection_frame, text=f"{var_name}:").grid(row=i, column=0, sticky='w', pady=5)
            
            var_combo = ttk.Combobox(selection_frame, values=available_columns, state="readonly")
            var_combo.grid(row=i, column=1, sticky='ew', padx=5, pady=5)
            
            # Set default selection if available
            if i < len(available_columns):
                var_combo.set(available_columns[i])
            
            selections[var_name] = var_combo
        
        # Configurar peso de la columna
        selection_frame.columnconfigure(1, weight=1)
        
        # Frame para botones
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        result = [None]  # Para almacenar el resultado
        
        def on_ok():
            mapping = {}
            for var_name, combo in selections.items():
                selected_col = combo.get()
                if not selected_col:
                    messagebox.showerror("Error", f"Please select a column for {var_name}", parent=dialog)
                    return
                mapping[var_name] = selected_col
            
            # Verificar que no hay columnas duplicadas
            if len(mapping.values()) != len(set(mapping.values())):
                messagebox.showerror("Error", "Column selections must be unique", parent=dialog)
                return
            
            result[0] = mapping
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        
        dialog.wait_window()
        
        if result[0] is not None:
            self.column_mapping[problem] = result[0]
            self._update_mode_display()
            messagebox.showinfo("Mapping Saved", f"Column mapping saved for {problem}")

    def start_training(self):
        """Iniciar entrenamiento"""
        try:
            # Verificar si hay mapeo de columnas para CSV
            problem = self.shared_state['problem_name'].get()
            csv_data = self.shared_state.get('current_dataframe')
            
            if csv_data is not None:
                if problem not in self.column_mapping:
                    response = messagebox.askyesno(
                        "CSV Data Available", 
                        "CSV data is loaded but no column mapping is configured. "
                        "Would you like to map columns now?\n\n"
                        "Click 'No' to use analytical solution instead."
                    )
                    if response:
                        self._prompt_column_mapping()
                        if problem not in self.column_mapping:
                            return  # User cancelled mapping
                    else:
                        # Use analytical solution
                        csv_data = None
            
            # Reset plots only when starting new training
            self.visualizer.reset_plots()
            self.visualizer.init_plots()
            self._update_plot_display()
            
            self._setup_trainer(csv_data)
            self._update_ui_for_training_start()
            self._training_loop()
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to start training:\n\n{e}")
            self.stop_training()

    def _setup_trainer(self, csv_data):
        """Configurar el entrenador PINN"""
        problem = self.shared_state['problem_name'].get()
        config = get_active_config(problem)
        
        # Obtener mapeo de columnas si estamos usando CSV
        column_mapping = None
        if csv_data is not None:
            column_mapping = self.column_mapping.get(problem)
            if not column_mapping:
                raise ValueError("Column mapping required for CSV data")
        
        self.shared_state['trainer'] = PINNTrainer(config, problem, csv_data, column_mapping)
        self.shared_state['is_training'] = True

    def _update_ui_for_training_start(self):
        """Actualizar UI para inicio de entrenamiento"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.problem_selector.config(state=tk.DISABLED)
        self.map_btn.config(state=tk.DISABLED)
        self.epoch = 0
        self.loss_history = []

    def stop_training(self):
        """Detener entrenamiento"""
        self.shared_state['is_training'] = False
        
        import tensorflow as tf
        tf.keras.backend.clear_session()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.problem_selector.config(state=tk.NORMAL)
        self.map_btn.config(state=tk.NORMAL)

    def _training_loop(self):
        """Bucle principal de entrenamiento"""
        if not self.shared_state.get('is_training', False):
            return

        try:
            trainer = self.shared_state['trainer']
            losses = trainer.perform_one_step()
            
            self.epoch += 1
            total_loss = losses[0].numpy()
            self.loss_history.append(total_loss)
            
            self._update_training_display(losses)
            
            self.after(50, self._training_loop)
            
        except Exception as e:
            self._handle_training_error(e)

    def _update_training_display(self, losses):
        """Actualizar visualización del entrenamiento"""
        if self.epoch % 10 == 0 or self.epoch == 1:
            self.epoch_label.config(text=f"Epoch: {self.epoch}")
            self.loss_label.config(text=f"Loss: {losses[0].numpy():.4e}")

        if self.epoch % 50 == 0:
            self.visualizer.update_loss_plot(self.epoch, self.loss_history)

        current_problem = self.shared_state['problem_name'].get()
        if current_problem == "HEAT":
            update_solution = (self.epoch == 1 or self.epoch % 100 == 0)
        else:
            update_solution = (self.epoch == 1 or self.epoch % 50 == 0)
            
        if update_solution:
            self._update_visualizations()

    def _update_visualizations(self):
        """Actualizar todas las visualizaciones"""
        trainer = self.shared_state['trainer']
        
        self.visualizer.update_loss_plot(self.epoch, self.loss_history)
        
        if hasattr(trainer, 'physics') and hasattr(trainer, 'model'):
            self._update_solution_visualization(trainer)
            self._update_metrics_summary(trainer)
        
        self._update_plot_display()

    def _update_solution_visualization(self, trainer):
        """Actualizar visualización de solución"""
        problem = trainer.active_problem
        
        if problem in ("SHO", "DHO"):
            self._visualize_ode_solution(trainer)
        elif problem == "HEAT":
            self._visualize_heat_solution(trainer)

    def _visualize_ode_solution(self, trainer):
        """Visualizar solución de ODE"""
        if trainer.physics.has_analytical:
            # Analytical mode
            t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
            t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
            
            x_pred = trainer.model(t_plot).numpy()
            x_true = trainer.physics.analytical_solution(t_plot)
            
            self.visualizer.update_solution_plot(t_plot, x_true, x_pred, "t", "x(t)")
            
            error = np.linalg.norm(x_pred - x_true) / (np.linalg.norm(x_true) + 1e-8)
            self.error_label.config(text=f"L2 Error: {error:.4f}")
        else:
            # CSV data mode
            t_data, x_true = trainer.physics.get_training_data()
            x_pred = trainer.model(t_data).numpy()
            
            self.visualizer.update_solution_plot(t_data, x_true, x_pred, "t", "x(t)")
            
            error = np.linalg.norm(x_pred - x_true) / (np.linalg.norm(x_true) + 1e-8)
            self.error_label.config(text=f"Data RMSE: {error:.4f}")

    def _visualize_heat_solution(self, trainer):
        """Visualizar solución de calor 2D"""
        if trainer.physics.has_analytical:
            # Analytical mode
            t_slice = 0.5
            y_slice = 0.5
            x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
            t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
            
            resolution = 30
            x_plot = np.linspace(x_domain[0], x_domain[1], resolution)
            t_plot = np.linspace(t_domain[0], t_domain[1], resolution)
            X, T = np.meshgrid(x_plot, t_plot)
            
            xy_flat = np.stack([
                X.flatten(), 
                np.full_like(X.flatten(), y_slice), 
                T.flatten()
            ], axis=1).astype(np.float32)
            
            u_pred = trainer.model(xy_flat).numpy().reshape(X.shape)
            
            self.visualizer.update_heat_solution_plot(
                X, T, u_pred, 
                f"Heat Equation (y={y_slice}) - Epoch {self.epoch}"
            )
            
            u_true = trainer.physics.analytical_solution(xy_flat).reshape(X.shape)
            error = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            self.error_label.config(text=f"L2 Error: {error:.4f}")
        else:
            # CSV data mode - show data points
            x_data, y_data, t_data, u_true = trainer.physics.get_training_data()
            
            # Create a simple 2D slice for visualization
            unique_y = np.unique(y_data)
            if len(unique_y) > 0:
                y_slice = unique_y[0]  # Use first y value
                mask = (y_data.flatten() == y_slice)
                x_slice = x_data[mask]
                t_slice = t_data[mask]
                u_slice = u_true[mask]
                
                if len(x_slice) > 0:
                    # Sort for better plotting
                    sort_idx = np.argsort(x_slice.flatten())
                    x_slice = x_slice[sort_idx]
                    t_slice = t_slice[sort_idx]
                    u_slice = u_slice[sort_idx]
                    
                    u_pred = trainer.model(
                        np.hstack([x_slice, np.full_like(x_slice, y_slice), t_slice])
                    ).numpy()
                    
                    # Use line plot for CSV data
                    self.visualizer.update_solution_plot(
                        x_slice, u_slice, u_pred, "x", f"Temperature (y={y_slice:.2f})"
                    )
                    
                    error = np.linalg.norm(u_pred - u_slice) / (np.linalg.norm(u_slice) + 1e-8)
                    self.error_label.config(text=f"Data RMSE: {error:.4f}")

    def _update_metrics_summary(self, trainer):
        """Actualizar resumen de métricas"""
        problem = trainer.active_problem
        
        if problem in ("SHO", "DHO"):
            if trainer.physics.has_analytical:
                t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
                t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
                y_pred = trainer.model(t_plot).numpy().flatten()
                y_true = trainer.physics.analytical_solution(t_plot).flatten()
                x_data = t_plot
            else:
                t_data, y_true_data = trainer.physics.get_training_data()
                y_pred = trainer.model(t_data).numpy().flatten()
                y_true = y_true_data.flatten()
                x_data = t_data.flatten()
        elif problem == "HEAT":
            if trainer.physics.has_analytical:
                x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
                t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
                y_slice = 0.5
                
                x_plot = np.linspace(x_domain[0], x_domain[1], 20)
                t_plot = np.linspace(t_domain[0], t_domain[1], 20)
                X, T = np.meshgrid(x_plot, t_plot)
                xy_flat = np.stack([X.flatten(), np.full_like(X.flatten(), y_slice), T.flatten()], axis=1)
                xy_flat = xy_flat.astype(np.float32)
                
                y_pred = trainer.model(xy_flat).numpy().flatten()
                y_true = trainer.physics.analytical_solution(xy_flat).flatten()
                x_data = X.flatten()
            else:
                x_data, y_data, t_data, u_true = trainer.physics.get_training_data()
                xy_flat = np.hstack([x_data, y_data, t_data])
                y_pred = trainer.model(xy_flat).numpy().flatten()
                y_true = u_true.flatten()
                x_data = x_data.flatten()

        metrics_report = self.metrics_calc.comprehensive_report(y_true, y_pred)
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        
        # Add mode information
        mode = "CSV Data" if trainer.use_csv else "Analytical"
        self.metrics_text.insert(tk.END, f"Training Mode: {mode}\n")
        self.metrics_text.insert(tk.END, f"Problem: {trainer.active_problem}\n")
        if trainer.column_mapping:
            self.metrics_text.insert(tk.END, f"Column Mapping: {trainer.column_mapping}\n")
        self.metrics_text.insert(tk.END, f"Epochs: {self.epoch}\n\n")
        self.metrics_text.insert(tk.END, metrics_report)
        self.metrics_text.config(state="disabled")
        
        self._update_report_tab(trainer, y_true, y_pred, x_data)

    def _update_report_tab(self, trainer, y_true, y_pred, x_data):
        """Actualiza la pestaña de reportes"""
        metrics_report = self.metrics_calc.comprehensive_report(y_true, y_pred)
        
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
        elif key == 'current_dataframe':
            self._update_mode_display()