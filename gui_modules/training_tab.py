# src/gui_modules/training_tab.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import threading

from src.training import PINNTrainer
from src.config import get_active_config

class TrainingTab(ttk.Frame):
    """
    Pestaña de Entrenamiento y Visualización.
    Permite modificar hiperparámetros de entrenamiento, arquitectura de red y física.
    """
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.report_tab_ref = None
        self.trainer = None
        
        # Almacén de variables de control
        self.param_vars = {} 
        
        self._setup_ui()

    def _setup_ui(self):
        # Layout principal: Panel Izquierdo (Controles) y Derecho (Gráfico)
        left_panel = ttk.Frame(self, padding=10)
        left_panel.pack(side="left", fill="y")
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side="right", fill="both", expand=True)

        # --- PANEL IZQUIERDO (Controles Scrollables si fuera necesario, pero fijo por ahora) ---
        ttk.Label(left_panel, text="Configuración del Sistema", font=("Helvetica", 12, "bold")).pack(pady=(0, 10))

        # 1. Selector de Problema
        ttk.Label(left_panel, text="Problema Físico:", font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.prob_combo = ttk.Combobox(left_panel, values=["SHO", "DHO", "HEAT"], state="readonly")
        self.prob_combo.set("SHO")
        self.prob_combo.pack(fill="x", pady=5)
        self.prob_combo.bind("<<ComboboxSelected>>", self._on_problem_change)

        # Frame contenedor de parámetros
        self.settings_canvas = tk.Canvas(left_panel, bd=0, highlightthickness=0)
        self.settings_frame = ttk.Frame(self.settings_canvas)
        self.settings_canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        self.settings_canvas.pack(side="top", fill="both", expand=True, pady=5)
        
        # Configurar el frame dinámico dentro
        self.settings_frame.bind("<Configure>", lambda e: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))

        # Botones de Control (Abajo del todo del panel izquierdo)
        control_frame = ttk.Frame(left_panel)
        control_frame.pack(side="bottom", fill="x", pady=10)
        
        self.btn_train = ttk.Button(control_frame, text="Start Training", command=self.toggle_training)
        self.btn_train.pack(fill="x", pady=5)
        
        self.lbl_epoch = ttk.Label(control_frame, text="Epoch: 0")
        self.lbl_epoch.pack()
        self.lbl_loss = ttk.Label(control_frame, text="Loss: ---")
        self.lbl_loss.pack()

        # --- PANEL DERECHO (Gráficos en vivo) ---
        self.fig = plt.figure(figsize=(6, 9))
        self.gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.5])
        
        self.ax_loss = self.fig.add_subplot(self.gs[0, :])      
        self.ax_slice = self.fig.add_subplot(self.gs[1, :])     
        self.ax_hm_pred = self.fig.add_subplot(self.gs[2, 0])   
        self.ax_hm_true = self.fig.add_subplot(self.gs[2, 1])   
        
        self.fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.95, bottom=0.05, left=0.12, right=0.95)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self._update_param_inputs()
        self._clear_plots()

    def _clear_plots(self):
        for ax in [self.ax_loss, self.ax_slice, self.ax_hm_pred, self.ax_hm_true]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.ax_loss.set_title("Esperando entrenamiento...")
        self.ax_hm_pred.axis('off')
        self.ax_hm_true.axis('off')
        self.canvas.draw()

    def _on_problem_change(self, event):
        self.shared_state['problem_name'].set(self.prob_combo.get())
        self._update_param_inputs()
        self._clear_plots()

    def _update_param_inputs(self):
        """Genera inputs dinámicos divididos en secciones."""
        # Limpiar widgets anteriores
        for widget in self.settings_frame.winfo_children():
            widget.destroy()
        self.param_vars = {}

        prob = self.prob_combo.get()
        config = get_active_config(prob)

        # --- SECCIÓN 1: ARQUITECTURA DEL MODELO ---
        self._add_section_header("Arquitectura de Red")
        model_conf = config.get("MODEL_CONFIG", {})
        
        # Capas Ocultas
        self._add_input_row("Capas Ocultas:", "mod_num_layers", model_conf.get("num_layers", 5), val_type=int)
        # Neuronas por capa
        self._add_input_row("Neuronas/Capa:", "mod_hidden_dim", model_conf.get("hidden_dim", 64), val_type=int)
        # Función de Activación
        activations = ["tanh", "sigmoid", "relu", "elu", "selu", "swish", "gelu"]
        current_act = model_conf.get("activation", "tanh")
        self._add_combo_row("Activación:", "mod_activation", activations, current_act)

        # --- SECCIÓN 2: ENTRENAMIENTO ---
        self._add_section_header("Entrenamiento")
        self._add_input_row("Learning Rate:", "train_lr", config.get("LEARNING_RATE", 1e-3), val_type=float)
        self._add_input_row("Epochs:", "train_epochs", config.get("EPOCHS", 5000), val_type=int)

        # --- SECCIÓN 3: PARÁMETROS FÍSICOS ---
        self._add_section_header("Parámetros Físicos")
        phys_conf = config.get("PHYSICS_CONFIG", {})
        for key, val in phys_conf.items():
            if isinstance(val, (int, float)): 
                self._add_input_row(f"{key}:", f"phys_{key}", val, val_type=type(val))

    def _add_section_header(self, text):
        lbl = ttk.Label(self.settings_frame, text=text, font=("Helvetica", 10, "bold", "underline"), foreground="#333")
        lbl.pack(fill="x", pady=(15, 5), padx=2)

    def _add_input_row(self, label_text, var_key, default_val, val_type=float):
        row = ttk.Frame(self.settings_frame)
        row.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(row, text=label_text, width=14, anchor="w").pack(side="left")
        
        if val_type is int:
            var = tk.IntVar(value=default_val)
        else:
            var = tk.DoubleVar(value=default_val)
            
        self.param_vars[var_key] = var
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="right", expand=True, fill="x")

    def _add_combo_row(self, label_text, var_key, options, default_val):
        row = ttk.Frame(self.settings_frame)
        row.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(row, text=label_text, width=14, anchor="w").pack(side="left")
        
        var = tk.StringVar(value=default_val)
        self.param_vars[var_key] = var
        
        combo = ttk.Combobox(row, textvariable=var, values=options, state="readonly")
        combo.pack(side="right", expand=True, fill="x")

    def toggle_training(self):
        if not self.shared_state['is_training']:
            self.start_training()
        else:
            self.stop_training()

    def start_training(self):
        self.shared_state['is_training'] = True
        self.btn_train.config(text="Stop Training")
        self._clear_plots()
        
        # 1. Cargar configuración base
        prob_name = self.prob_combo.get()
        base_config = get_active_config(prob_name)
        
        try:
            # 2. Inyectar Hiperparámetros de Modelo (NUEVO)
            if "MODEL_CONFIG" not in base_config: base_config["MODEL_CONFIG"] = {}
            
            base_config["MODEL_CONFIG"]["num_layers"] = self.param_vars["mod_num_layers"].get()
            base_config["MODEL_CONFIG"]["hidden_dim"] = self.param_vars["mod_hidden_dim"].get()
            base_config["MODEL_CONFIG"]["activation"] = self.param_vars["mod_activation"].get()

            # 3. Inyectar Hiperparámetros de Entrenamiento
            base_config["LEARNING_RATE"] = self.param_vars["train_lr"].get()
            base_config["EPOCHS"] = self.param_vars["train_epochs"].get()

            # 4. Inyectar Parámetros Físicos
            for key, var in self.param_vars.items():
                if key.startswith("phys_"):
                    phys_key = key.replace("phys_", "")
                    base_config["PHYSICS_CONFIG"][phys_key] = var.get()
                    
        except Exception as e:
            print(f"Error leyendo parámetros de la GUI: {e}")
            self.stop_training()
            return

        # Iniciar Trainer
        self.trainer = PINNTrainer(base_config, prob_name)
        
        # Cargar datos externos si existen
        if self.shared_state.get('current_dataframe') is not None:
            path = self.shared_state.get('external_data_path')
            if path:
                self.trainer.data_manager.load_external_data(path)
                self.trainer.data_manager.prepare_data()

        self.shared_state['trainer'] = self.trainer
        
        # Hilo de entrenamiento
        threading.Thread(target=self._training_loop, daemon=True).start()

    def stop_training(self):
        self.shared_state['is_training'] = False
        self.btn_train.config(text="Start Training")

    def _training_loop(self):
        epochs_target = self.trainer.config["EPOCHS"]
        
        while self.shared_state['is_training']:
            losses = self.trainer.perform_one_step()
            epoch = self.trainer.epoch
            current_loss = losses[0].numpy()
            
            if epoch % 10 == 0:
                self.after(0, self._update_gui_metrics, epoch, current_loss)
            
            if epoch >= epochs_target:
                self.stop_training()
                break

    def _update_gui_metrics(self, epoch, loss):
        """Actualiza las gráficas."""
        self.lbl_epoch.config(text=f"Epoch: {epoch}")
        self.lbl_loss.config(text=f"Loss: {loss:.2e}")
        
        prob_name = self.shared_state['problem_name'].get()
        
        # 1. Gráfico de Loss
        self.ax_loss.clear()
        hist = self.trainer.loss_history
        plot_hist = hist if len(hist) < 2000 else hist[-2000:]
        self.ax_loss.plot(plot_hist, label='Total Loss', color='#1f77b4')
        self.ax_loss.set_yscale('log')
        self.ax_loss.set_title("Convergencia")
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.legend(loc='upper right', fontsize='x-small')

        # 2. Visualización Específica
        try:
            physics = self.trainer.physics
            model = self.trainer.model
            config = self.trainer.config["PHYSICS_CONFIG"]
            
            self.ax_slice.clear()
            
            if prob_name in ["SHO", "DHO"]:
                self.ax_hm_pred.axis('off')
                self.ax_hm_true.axis('off')
                
                t_min, t_max = config["t_domain"]
                t_grid = np.linspace(t_min, t_max, 200).reshape(-1, 1).astype(np.float32)
                t_tf = tf.convert_to_tensor(t_grid)
                
                u_pred = model(t_tf).numpy()
                u_true = physics.analytical_solution(t_grid)
                
                self.ax_slice.plot(t_grid, u_true, 'k--', label='Analítica')
                self.ax_slice.plot(t_grid, u_pred, 'r-', alpha=0.8, label='PINN')
                self.ax_slice.set_title(f"Dinámica ({prob_name})")
                self.ax_slice.legend(fontsize='small')
                self.ax_slice.grid(True)

            elif prob_name == "HEAT":
                x_min, x_max = config["x_domain"]
                y_min, y_max = config["y_domain"]
                t_mid = sum(config["t_domain"]) / 2
                y_mid = sum(config["y_domain"]) / 2
                
                # Gráfico Medio: Slice
                x_grid = np.linspace(x_min, x_max, 100).reshape(-1, 1).astype(np.float32)
                y_grid_1d = np.full_like(x_grid, y_mid)
                t_grid_1d = np.full_like(x_grid, t_mid)
                
                xyt_slice = np.hstack([x_grid, y_grid_1d, t_grid_1d])
                xyt_tf = tf.convert_to_tensor(xyt_slice)
                
                u_slice_pred = model(xyt_tf).numpy()
                u_slice_true = physics.analytical_solution(xyt_slice)
                
                self.ax_slice.plot(x_grid, u_slice_true, 'k--', label='Real')
                self.ax_slice.plot(x_grid, u_slice_pred, 'r-', alpha=0.8, label='PINN')
                self.ax_slice.set_title(f"Corte y={y_mid:.1f}, t={t_mid:.2f}")
                self.ax_slice.legend(fontsize='x-small')
                self.ax_slice.grid(True)

                # Heatmaps
                N = 50
                x = np.linspace(x_min, x_max, N)
                y = np.linspace(y_min, y_max, N)
                X, Y = np.meshgrid(x, y)
                T_flat = np.full_like(X.flatten(), t_mid)
                
                input_2d = np.stack([X.flatten(), Y.flatten(), T_flat], axis=1).astype(np.float32)
                input_tf = tf.convert_to_tensor(input_2d)
                
                Z_pred = model(input_tf).numpy().reshape(N, N)
                Z_true = physics.analytical_solution(input_2d).reshape(N, N)
                
                vmin, vmax = min(Z_true.min(), Z_pred.min()), max(Z_true.max(), Z_pred.max())
                
                self.ax_hm_pred.axis('on')
                self.ax_hm_pred.imshow(Z_pred, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                self.ax_hm_pred.set_title("Predicción")
                self.ax_hm_pred.set_xticks([]) # Limpiar ejes
                self.ax_hm_pred.set_yticks([])

                self.ax_hm_true.axis('on')
                self.ax_hm_true.imshow(Z_true, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                self.ax_hm_true.set_title("Analítica")
                self.ax_hm_true.set_xticks([])
                self.ax_hm_true.set_yticks([])

        except Exception as e:
            print(f"Error visualización: {e}")

        self.canvas.draw()

    def on_shared_state_change(self, key, value):
        pass