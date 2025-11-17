#gui_modules/training_tab.py
"""
Módulo para la pestaña de entrenamiento - PROPER DIMENSIONS WITH PLOTLY
(Versión 0.0.5 - Con botón de modo analítico y mejoras CSV)
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


class HyperparameterDialog(tk.Toplevel):
    """Diálogo para editar hiperparámetros del entrenamiento"""
    
    def __init__(self, parent, config, problem_name):
        super().__init__(parent)
        self.title(f"Hyperparameters - {problem_name}")
        self.geometry("500x600")
        self.transient(parent)
        self.grab_set()
        
        self.config = config
        self.problem_name = problem_name
        self.result = None
        
        self._build_ui()
        
    def _build_ui(self):
        """Construir la interfaz del diálogo"""
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text=f"Hyperparameters for {self.problem_name}", 
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Frame con scroll
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Hiperparámetros básicos
        self._build_basic_params(scrollable_frame)
        
        # Pesos de pérdida
        self._build_loss_weights(scrollable_frame)
        
        # Configuración de física
        self._build_physics_params(scrollable_frame)
        
        # Configuración de datos
        self._build_data_params(scrollable_frame)
        
        # Botones
        self._build_buttons(main_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _build_basic_params(self, parent):
        """Construir sección de parámetros básicos"""
        frame = ttk.Labelframe(parent, text="Basic Training Parameters", padding="10")
        frame.pack(fill=tk.X, pady=5)
        
        # Learning Rate
        ttk.Label(frame, text="Learning Rate:").pack(anchor='w', pady=2)
        self.lr_var = tk.StringVar(value=str(self.config['LEARNING_RATE']))
        lr_entry = ttk.Entry(frame, textvariable=self.lr_var, width=15)
        lr_entry.pack(anchor='w', pady=2)
        ttk.Label(frame, text="(e.g., 1e-3, 5e-4)").pack(anchor='w', pady=2)
        
        # Epochs
        ttk.Label(frame, text="Epochs:").pack(anchor='w', pady=2)
        self.epochs_var = tk.StringVar(value=str(self.config['EPOCHS']))
        epochs_entry = ttk.Entry(frame, textvariable=self.epochs_var, width=15)
        epochs_entry.pack(anchor='w', pady=2)
        
        # Hidden Dimensions
        ttk.Label(frame, text="Hidden Dimensions:").pack(anchor='w', pady=2)
        self.hidden_dim_var = tk.StringVar(value=str(self.config['MODEL_CONFIG']['hidden_dim']))
        hidden_dim_entry = ttk.Entry(frame, textvariable=self.hidden_dim_var, width=15)
        hidden_dim_entry.pack(anchor='w', pady=2)
        
        # Number of Layers
        ttk.Label(frame, text="Number of Layers:").pack(anchor='w', pady=2)
        self.num_layers_var = tk.StringVar(value=str(self.config['MODEL_CONFIG']['num_layers']))
        num_layers_entry = ttk.Entry(frame, textvariable=self.num_layers_var, width=15)
        num_layers_entry.pack(anchor='w', pady=2)
        
        # Activation Function
        ttk.Label(frame, text="Activation:").pack(anchor='w', pady=2)
        self.activation_var = tk.StringVar(value=self.config['MODEL_CONFIG']['activation'])
        activation_combo = ttk.Combobox(frame, textvariable=self.activation_var, 
                                      values=['tanh', 'relu', 'sigmoid', 'elu'], state="readonly", width=13)
        activation_combo.pack(anchor='w', pady=2)
        
    def _build_loss_weights(self, parent):
        """Construir sección de pesos de pérdida"""
        frame = ttk.Labelframe(parent, text="Loss Weights", padding="10")
        frame.pack(fill=tk.X, pady=5)
        
        self.loss_weight_vars = {}
        
        for loss_name, loss_weight in self.config['LOSS_WEIGHTS'].items():
            ttk.Label(frame, text=f"{loss_name}:").pack(anchor='w', pady=2)
            var = tk.StringVar(value=str(loss_weight))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.pack(anchor='w', pady=2)
            self.loss_weight_vars[loss_name] = var
            
    def _build_physics_params(self, parent):
        """Construir sección de parámetros de física"""
        frame = ttk.Labelframe(parent, text="Physics Parameters", padding="10")
        frame.pack(fill=tk.X, pady=5)
        
        self.physics_vars = {}
        
        physics_config = self.config['PHYSICS_CONFIG']
        
        for param_name, param_value in physics_config.items():
            if param_name == 'initial_conditions':
                continue  # Manejar por separado
                
            ttk.Label(frame, text=f"{param_name}:").pack(anchor='w', pady=2)
            
            # FIXED: Handle domain parameters (lists) properly
            if param_name.endswith('_domain') and isinstance(param_value, list):
                str_value = str(param_value)  # This will be like "[0.0, 2.0]"
            else:
                str_value = str(param_value)
                
            var = tk.StringVar(value=str_value)
            entry = ttk.Entry(frame, textvariable=var, width=15)
            entry.pack(anchor='w', pady=2)
            self.physics_vars[param_name] = var
            
        # Condiciones iniciales
        if 'initial_conditions' in physics_config:
            ic_frame = ttk.Frame(frame)
            ic_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(ic_frame, text="Initial Conditions:", font=('Arial', 9, 'bold')).pack(anchor='w')
            
            ic_config = physics_config['initial_conditions']
            self.ic_vars = {}
            
            for ic_name, ic_value in ic_config.items():
                ic_row_frame = ttk.Frame(ic_frame)
                ic_row_frame.pack(fill=tk.X, pady=1)
                
                ttk.Label(ic_row_frame, text=f"  {ic_name}:").pack(side=tk.LEFT, padx=(10, 5))
                var = tk.StringVar(value=str(ic_value))
                entry = ttk.Entry(ic_row_frame, textvariable=var, width=10)
                entry.pack(side=tk.LEFT, padx=5)
                self.ic_vars[ic_name] = var

    def _build_data_params(self, parent):
        """Construir sección de parámetros de datos"""
        frame = ttk.Labelframe(parent, text="Data Sampling", padding="10")
        frame.pack(fill=tk.X, pady=5)
        
        self.data_vars = {}
        
        data_config = self.config['DATA_CONFIG']
        
        for param_name, param_value in data_config.items():
            ttk.Label(frame, text=f"{param_name}:").pack(anchor='w', pady=2)
            var = tk.StringVar(value=str(param_value))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.pack(anchor='w', pady=2)
            self.data_vars[param_name] = var
            
    def _build_buttons(self, parent):
        """Construir botones de acción"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Apply", command=self._on_apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self._on_reset).pack(side=tk.LEFT, padx=5)
        
    def _on_apply(self):
        """Aplicar cambios sin cerrar"""
        if self._validate_inputs():
            self._collect_results()
            messagebox.showinfo("Success", "Hyperparameters applied successfully!")
        
    def _on_ok(self):
        """Aplicar cambios y cerrar"""
        if self._validate_inputs():
            self._collect_results()
            self.destroy()
        
    def _on_cancel(self):
        """Cancelar sin aplicar cambios"""
        self.result = None
        self.destroy()
        
    def _on_reset(self):
        """Restablecer valores por defecto"""
        from src.config import get_active_config
        default_config = get_active_config(self.problem_name)
        
        # Restablecer todos los valores
        self.lr_var.set(str(default_config['LEARNING_RATE']))
        self.epochs_var.set(str(default_config['EPOCHS']))
        self.hidden_dim_var.set(str(default_config['MODEL_CONFIG']['hidden_dim']))
        self.num_layers_var.set(str(default_config['MODEL_CONFIG']['num_layers']))
        self.activation_var.set(default_config['MODEL_CONFIG']['activation'])
        
        # Restablecer pesos de pérdida
        for loss_name, var in self.loss_weight_vars.items():
            if loss_name in default_config['LOSS_WEIGHTS']:
                var.set(str(default_config['LOSS_WEIGHTS'][loss_name]))
                
        # Restablecer parámetros de física
        for param_name, var in self.physics_vars.items():
            if param_name in default_config['PHYSICS_CONFIG']:
                var.set(str(default_config['PHYSICS_CONFIG'][param_name]))
                
        # Restablecer condiciones iniciales
        if hasattr(self, 'ic_vars'):
            for ic_name, var in self.ic_vars.items():
                if (ic_name in default_config['PHYSICS_CONFIG'].get('initial_conditions', {})):
                    var.set(str(default_config['PHYSICS_CONFIG']['initial_conditions'][ic_name]))
                    
        # Restablecer parámetros de datos
        for param_name, var in self.data_vars.items():
            if param_name in default_config['DATA_CONFIG']:
                var.set(str(default_config['DATA_CONFIG'][param_name]))
        
    def _validate_inputs(self):
        """Validar las entradas del usuario"""
        try:
            # Validar números básicos
            float(self.lr_var.get())
            int(self.epochs_var.get())
            int(self.hidden_dim_var.get())
            int(self.num_layers_var.get())
            
            # Validar pesos de pérdida
            for var in self.loss_weight_vars.values():
                float(var.get())
                
            # Validar parámetros de física - FIXED: Handle domain lists
            for param_name, var in self.physics_vars.items():
                value = var.get()
                # Skip domain parameters (they are lists, not floats)
                if param_name.endswith('_domain'):
                    continue  # Domain parameters are handled separately
                float(value)
                
            # Validar condiciones iniciales
            if hasattr(self, 'ic_vars'):
                for var in self.ic_vars.values():
                    float(var.get())
                    
            # Validar parámetros de datos - FIXED: Iterate over values, not items
            for var in self.data_vars.values():  # Changed from .items() to .values()
                int(var.get())
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Invalid input format:\n\n{e}")
            return False

    def _collect_results(self):
        """Recolectar resultados validados"""
        self.result = {
            'learning_rate': float(self.lr_var.get()),
            'epochs': int(self.epochs_var.get()),
            'model_config': {
                'hidden_dim': int(self.hidden_dim_var.get()),
                'num_layers': int(self.num_layers_var.get()),
                'activation': self.activation_var.get()
            },
            'loss_weights': {k: float(v.get()) for k, v in self.loss_weight_vars.items()},
            'physics_config': {},
            'data_config': {k: int(v.get()) for k, v in self.data_vars.items()}
        }
        
        # Handle physics parameters - FIXED: Handle domain parameters
        for param_name, var in self.physics_vars.items():
            value = var.get()
            if param_name.endswith('_domain'):
                # Domain parameters are lists like "[0.0, 2.0]"
                try:
                    # Remove brackets and split by comma
                    clean_value = value.strip('[]')
                    parts = clean_value.split(',')
                    if len(parts) == 2:
                        domain_list = [float(parts[0].strip()), float(parts[1].strip())]
                        self.result['physics_config'][param_name] = domain_list
                    else:
                        # Use default if parsing fails
                        default_config = get_active_config(self.problem_name)
                        self.result['physics_config'][param_name] = default_config['PHYSICS_CONFIG'][param_name]
                except:
                    # Use default if parsing fails
                    default_config = get_active_config(self.problem_name)
                    self.result['physics_config'][param_name] = default_config['PHYSICS_CONFIG'][param_name]
            else:
                # Regular float parameters
                self.result['physics_config'][param_name] = float(value)
        
        # Añadir condiciones iniciales si existen
        if hasattr(self, 'ic_vars'):
            self.result['physics_config']['initial_conditions'] = {
                k: float(v.get()) for k, v in self.ic_vars.items()
            }

class TrainingTab(ttk.Frame):
    """Pestaña de entrenamiento con botón de modo analítico y mejoras CSV"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.visualizer = TrainingVisualizer()
        self.metrics_calc = MetricsCalculator()
        self.report_tab_ref = None
        
        self.plot_label = None
        self.column_mapping = {}  # Almacena los mapeos por problema
        self.custom_configs = {}  # Almacena configuraciones personalizadas por problema
        
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

        # NEW: Botón para volver a modo analítico
        self.analytical_btn = ttk.Button(control_frame, text="Switch to Analytical Mode", 
                                        command=self._switch_to_analytical_mode)
        self.analytical_btn.pack(pady=5, fill=tk.X)

        # Botón para hiperparámetros
        self.hyperparam_btn = ttk.Button(control_frame, text="Hyperparameters...", 
                                        command=self._open_hyperparameter_dialog)
        self.hyperparam_btn.pack(pady=5, fill=tk.X)

        # Indicador de modo
        self.mode_label = ttk.Label(control_frame, text="Mode: Analytical", 
                                   font=('Arial', 9, 'bold'), foreground='blue')
        self.mode_label.pack(pady=5)

        # Información de mapeo
        self.mapping_info = ttk.Label(control_frame, text="No CSV mapping", 
                                     font=('Arial', 8), foreground='gray', wraplength=200)
        self.mapping_info.pack(pady=5)

        # Información de hiperparámetros
        self.hyperparam_info = ttk.Label(control_frame, text="Default config", 
                                        font=('Arial', 8), foreground='green', wraplength=200)
        self.hyperparam_info.pack(pady=5)

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
        
        # Actualizar información de modo
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
        
        # Actualizar información de hiperparámetros
        if problem in self.custom_configs:
            self.hyperparam_info.config(text="Custom config", foreground='orange')
        else:
            self.hyperparam_info.config(text="Default config", foreground='green')

    def _switch_to_analytical_mode(self):
        """Switch from CSV mode to analytical mode"""
        problem = self.shared_state['problem_name'].get()
        
        if problem in self.column_mapping:
            # Clear CSV mapping for this problem
            del self.column_mapping[problem]
            self._update_mode_display()
            messagebox.showinfo("Mode Changed", f"Switched to analytical mode for {problem}")
            
            # If currently training with CSV data, stop training
            if self.shared_state.get('is_training', False):
                self.stop_training()
        else:
            messagebox.showinfo("Already Analytical", "Already in analytical mode")

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

    def _open_hyperparameter_dialog(self):
        """Abrir diálogo de hiperparámetros"""
        problem = self.shared_state['problem_name'].get()
        
        # Usar configuración personalizada si existe, sino la por defecto
        if problem in self.custom_configs:
            config = self.custom_configs[problem]
        else:
            config = get_active_config(problem)
        
        dialog = HyperparameterDialog(self, config, problem)
        self.wait_window(dialog)
        
        if dialog.result is not None:
            # Guardar configuración personalizada
            self.custom_configs[problem] = self._apply_hyperparameter_changes(problem, dialog.result)
            self._update_mode_display()
            messagebox.showinfo("Success", f"Hyperparameters updated for {problem}")

    def _apply_hyperparameter_changes(self, problem, new_params):
        """Aplicar cambios de hiperparámetros a la configuración"""
        config = get_active_config(problem).copy()  # Copia de la configuración por defecto
        
        # Actualizar parámetros básicos
        config['LEARNING_RATE'] = new_params['learning_rate']
        config['EPOCHS'] = new_params['epochs']
        
        # Actualizar configuración del modelo
        config['MODEL_CONFIG'].update(new_params['model_config'])
        
        # Actualizar pesos de pérdida
        config['LOSS_WEIGHTS'].update(new_params['loss_weights'])
        
        # Actualizar parámetros de física
        config['PHYSICS_CONFIG'].update(new_params['physics_config'])
        
        # Actualizar configuración de datos
        config['DATA_CONFIG'].update(new_params['data_config'])
        
        return config

    def start_training(self):
        """Iniciar entrenamiento"""
        try:
            # Clear any previous training state
            self.shared_state['is_training'] = False
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
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
            
            # Reset training state
            self.epoch = 0
            self.loss_history = []
            
            print(f" Starting training for {problem}...")
            self._setup_trainer(csv_data)
            self._update_ui_for_training_start()
            
            # Start training loop
            self._training_loop()
            
        except Exception as e:
            error_msg = f"Failed to start training:\n\n{str(e)}"
            print(f" {error_msg}")
            messagebox.showerror("Training Error", error_msg)
            self.stop_training()

    def _setup_trainer(self, csv_data):
        """Configurar el entrenador PINN"""
        problem = self.shared_state['problem_name'].get()
        
        # Usar configuración personalizada si existe
        if problem in self.custom_configs:
            config = self.custom_configs[problem]
        else:
            config = get_active_config(problem)
        
        # Obtener mapeo de columnas si estamos usando CSV
        column_mapping = None
        if csv_data is not None:
            column_mapping = self.column_mapping.get(problem)
            if not column_mapping:
                raise ValueError("Column mapping required for CSV data")
        
        print(f" Setting up trainer: Problem={problem}, Mode={'CSV' if csv_data is not None else 'Analytical'}")
        self.shared_state['trainer'] = PINNTrainer(config, problem, csv_data, column_mapping)
        self.shared_state['is_training'] = True
        print(" Trainer setup complete")

    def _update_ui_for_training_start(self):
        """Actualizar UI para inicio de entrenamiento"""
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.problem_selector.config(state=tk.DISABLED)
        self.map_btn.config(state=tk.DISABLED)
        self.analytical_btn.config(state=tk.DISABLED)
        self.hyperparam_btn.config(state=tk.DISABLED)
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
        self.analytical_btn.config(state=tk.NORMAL)
        self.hyperparam_btn.config(state=tk.NORMAL)

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
            x_true = trainer.physics.analytical_solution(t_plot).numpy()
            
            self.visualizer.update_solution_plot(t_plot, x_true, x_pred, "t", "x(t)")
            
            error = np.linalg.norm(x_pred - x_true) / (np.linalg.norm(x_true) + 1e-8)
            self.error_label.config(text=f"L2 Error: {error:.4f}")
        else:
            # CSV data mode - FIXED: Better visualization for CSV data
            t_data, x_true = trainer.physics.get_training_data()
            x_pred = trainer.model(t_data).numpy()
            
            # Sort by time for better plotting
            # FIXED: Use reshape instead of flatten for numpy arrays
            sort_idx = np.argsort(t_data.reshape(-1))
            t_sorted = t_data[sort_idx]
            x_true_sorted = x_true[sort_idx]
            x_pred_sorted = x_pred[sort_idx]
            
            self.visualizer.update_solution_plot(t_sorted, x_true_sorted, x_pred_sorted, "t", "x(t)")
            
            mse = np.mean((x_pred - x_true) ** 2)
            self.error_label.config(text=f"Data MSE: {mse:.4e}")    

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
                X.reshape(-1),  # FIXED: Use reshape instead of flatten
                np.full_like(X.reshape(-1), y_slice),  # FIXED: Use reshape instead of flatten
                T.reshape(-1)  # FIXED: Use reshape instead of flatten
            ], axis=1).astype(np.float32)
            
            u_pred = trainer.model(xy_flat).numpy().reshape(X.shape)
            
            self.visualizer.update_heat_solution_plot(
                X, T, u_pred, 
                f"Heat Equation (y={y_slice}) - Epoch {self.epoch}"
            )
            
            u_true = trainer.physics.analytical_solution(xy_flat).numpy().reshape(X.shape)
            error = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            self.error_label.config(text=f"L2 Error: {error:.4f}")
        else:
            # CSV data mode - show data points
            x_data, y_data, t_data, u_true = trainer.physics.get_training_data()
            
            # Create a simple 2D slice for visualization
            unique_y = np.unique(y_data)
            if len(unique_y) > 0:
                y_slice = unique_y[0]  # Use first y value
                mask = (y_data.reshape(-1) == y_slice)  # FIXED: Use reshape instead of flatten
                x_slice = x_data[mask]
                t_slice = t_data[mask]
                u_slice = u_true[mask]
                
                if len(x_slice) > 0:
                    # Sort for better plotting
                    # FIXED: Use reshape instead of flatten
                    sort_idx = np.argsort(x_slice.reshape(-1))
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
                    
                    mse = np.mean((u_pred - u_slice) ** 2)
                    self.error_label.config(text=f"Data MSE: {mse:.4e}")

    def _update_metrics_summary(self, trainer):
        """Actualizar resumen de métricas"""
        problem = trainer.active_problem
        
        if problem in ("SHO", "DHO"):
            if trainer.physics.has_analytical:
                t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
                t_plot = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
                y_pred = trainer.model(t_plot).numpy()
                y_true = trainer.physics.analytical_solution(t_plot).numpy()
                # FIXED: Use proper flattening for numpy arrays
                y_pred_flat = y_pred.reshape(-1)
                y_true_flat = y_true.reshape(-1)
                x_data = t_plot.reshape(-1)
            else:
                t_data, y_true_data = trainer.physics.get_training_data()
                y_pred = trainer.model(t_data).numpy()
                y_true = y_true_data
                # FIXED: Use proper flattening for numpy arrays
                y_pred_flat = y_pred.reshape(-1)
                y_true_flat = y_true.reshape(-1)
                x_data = t_data.reshape(-1)
        elif problem == "HEAT":
            if trainer.physics.has_analytical:
                x_domain = trainer.config["PHYSICS_CONFIG"]["x_domain"]
                t_domain = trainer.config["PHYSICS_CONFIG"]["t_domain"]
                y_slice = 0.5
                
                x_plot = np.linspace(x_domain[0], x_domain[1], 20)
                t_plot = np.linspace(t_domain[0], t_domain[1], 20)
                X, T = np.meshgrid(x_plot, t_plot)
                xy_flat = np.stack([X.reshape(-1), np.full_like(X.reshape(-1), y_slice), T.reshape(-1)], axis=1)
                xy_flat = xy_flat.astype(np.float32)
                
                y_pred = trainer.model(xy_flat).numpy()
                y_true = trainer.physics.analytical_solution(xy_flat).numpy()
                # FIXED: Use proper flattening for numpy arrays
                y_pred_flat = y_pred.reshape(-1)
                y_true_flat = y_true.reshape(-1)
                x_data = X.reshape(-1)
            else:
                x_data, y_data, t_data, u_true = trainer.physics.get_training_data()
                xy_flat = np.hstack([x_data, y_data, t_data])
                y_pred = trainer.model(xy_flat).numpy()
                y_true = u_true
                # FIXED: Use proper flattening for numpy arrays
                y_pred_flat = y_pred.reshape(-1)
                y_true_flat = y_true.reshape(-1)
                x_data = x_data.reshape(-1)

        metrics_report = self.metrics_calc.comprehensive_report(y_true_flat, y_pred_flat)
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        
        # Add mode information
        mode = "CSV Data" if trainer.use_csv else "Analytical"
        self.metrics_text.insert(tk.END, f"Training Mode: {mode}\n")
        self.metrics_text.insert(tk.END, f"Problem: {trainer.active_problem}\n")
        if trainer.column_mapping:
            self.metrics_text.insert(tk.END, f"Column Mapping: {trainer.column_mapping}\n")
        self.metrics_text.insert(tk.END, f"Epochs: {self.epoch}\n")
        self.metrics_text.insert(tk.END, f"Learning Rate: {trainer.config['LEARNING_RATE']}\n")
        self.metrics_text.insert(tk.END, f"Hidden Dim: {trainer.config['MODEL_CONFIG']['hidden_dim']}\n")
        self.metrics_text.insert(tk.END, f"Layers: {trainer.config['MODEL_CONFIG']['num_layers']}\n\n")
        self.metrics_text.insert(tk.END, metrics_report)
        self.metrics_text.config(state="disabled")
        
        self._update_report_tab(trainer, y_true_flat, y_pred_flat, x_data)

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