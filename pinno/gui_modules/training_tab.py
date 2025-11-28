# pinno/gui_modules/training_tab.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import threading

from ..training import PINNTrainer
from ..config import get_active_config, get_problem_variables

class TrainingTab(ttk.Frame):
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.trainer = None
        self.param_vars = {}
        
        # Inicializar default
        self._init_default_parameters("SHO")
        self._setup_ui()

    def _setup_ui(self):
        # --- LAYOUT PRINCIPAL ---
        left_panel = ttk.Frame(self, padding=10, width=320)
        left_panel.pack(side="left", fill="y")
        
        right_panel = ttk.Frame(self)
        right_panel.pack(side="right", fill="both", expand=True)

        # ==========================================
        # PANEL IZQUIERDO (CONTROLES)
        # ==========================================
        
        # 1. CONFIGURACION
        ttk.Label(left_panel, text="1. Problem Setup", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0,5))
        self.prob_combo = ttk.Combobox(left_panel, values=["SHO", "DHO", "HEAT"], state="readonly")
        self.prob_combo.set("SHO")
        self.prob_combo.pack(fill="x", pady=5)
        self.prob_combo.bind("<<ComboboxSelected>>", self._on_problem_change)

        # 2. MODO DE DATOS
        mode_group = ttk.LabelFrame(left_panel, text="2. Data Source", padding=10)
        mode_group.pack(fill="x", pady=15)
        
        self.btn_switch_mode = ttk.Button(mode_group, text="Switch to CSV Mode", command=self._toggle_mode)
        self.btn_switch_mode.pack(fill="x", pady=5)
        
        self.btn_map = ttk.Button(mode_group, text="🗺️ Map CSV Columns...", 
                                command=self._open_mapping_dialog, state="disabled")
        self.btn_map.pack(fill="x", pady=(5,0))

        self.lbl_status_mode = ttk.Label(mode_group, text="Current: Analytical", foreground="blue", font=("Arial", 9))
        self.lbl_status_mode.pack(anchor="w", pady=(10,0))
        self.lbl_csv_check = ttk.Label(mode_group, text="CSV Loaded: [No]", foreground="gray", font=("Arial", 8))
        self.lbl_csv_check.pack(anchor="w")

        # 3. HIPERPARAMETROS
        ttk.Label(left_panel, text="3. Configuration", font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(15,5))
        self.btn_params = ttk.Button(left_panel, text="⚙️ Hyperparameters...", command=self._open_params_window)
        self.btn_params.pack(fill="x", pady=5, ipady=5)

        # 4. ACCIONES
        self.btn_train = ttk.Button(left_panel, text="▶ START TRAINING", command=self.toggle_training)
        self.btn_train.pack(fill="x", pady=30, ipady=10)

        # METRICAS
        metrics_group = ttk.LabelFrame(left_panel, text="Live Metrics", padding=10)
        metrics_group.pack(fill="x", side="bottom", pady=10)
        self.lbl_epoch = ttk.Label(metrics_group, text="Epoch: 0")
        self.lbl_epoch.pack(anchor="w")
        self.lbl_loss = ttk.Label(metrics_group, text="Loss: ---")
        self.lbl_loss.pack(anchor="w")

        # ==========================================
        # PANEL DERECHO (GRAFICOS COMPLEX)
        # ==========================================
        self.fig = plt.figure(figsize=(8, 9))
        self.gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.5])
        
        self.ax_loss = self.fig.add_subplot(self.gs[0, :])
        self.ax_slice = self.fig.add_subplot(self.gs[1, :])
        self.ax_hm_pred = self.fig.add_subplot(self.gs[2, 0])
        self.ax_hm_true = self.fig.add_subplot(self.gs[2, 1])
        
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._reset_plots()
        self._refresh_ui_state()

    def _reset_plots(self):
        for ax in [self.ax_loss, self.ax_slice, self.ax_hm_pred, self.ax_hm_true]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        self.ax_loss.set_title("Loss History")
        self.ax_slice.set_title("Solution Slice")
        self.ax_hm_pred.axis('off')
        self.ax_hm_true.axis('off')
        self.canvas.draw()

    # --- PARAMETERS LOGIC ---
    def _init_default_parameters(self, problem_name):
        """Carga parametros iniciales, incluyendo Condiciones Iniciales anidadas."""
        config = get_active_config(problem_name)
        self.param_vars = {}
        
        # Modelo
        mc = config.get("MODEL_CONFIG", {})
        self.param_vars["mod_num_layers"] = tk.IntVar(value=mc.get("num_layers", 5))
        self.param_vars["mod_hidden_dim"] = tk.IntVar(value=mc.get("hidden_dim", 64))
        self.param_vars["mod_activation"] = tk.StringVar(value=mc.get("activation", "tanh"))
        
        # Training
        self.param_vars["EPOCHS"] = tk.IntVar(value=config.get("EPOCHS", 5000))
        self.param_vars["LEARNING_RATE"] = tk.DoubleVar(value=config.get("LEARNING_RATE", 1e-3))
        
        # Fisica (Scalar params)
        pc = config.get("PHYSICS_CONFIG", {})
        for k, v in pc.items():
            if isinstance(v, (int, float)):
                val_type = tk.IntVar if isinstance(v, int) else tk.DoubleVar
                self.param_vars[f"phys_{k}"] = val_type(value=v)
                
        # [NUEVO] Condiciones Iniciales (Diccionario anidado)
        if "initial_conditions" in pc and isinstance(pc["initial_conditions"], dict):
            for ic_k, ic_v in pc["initial_conditions"].items():
                self.param_vars[f"phys_IC_{ic_k}"] = tk.DoubleVar(value=ic_v)

    def _open_params_window(self):
        win = tk.Toplevel(self)
        win.title("Advanced Hyperparameters")
        win.geometry("400x600") # Un poco mas alto para que quepan las ICs
        win.transient(self)
        win.grab_set()
        
        main_f = ttk.Frame(win, padding=15)
        main_f.pack(fill="both", expand=True)
        
        # 1. Neural Network
        ttk.Label(main_f, text="Neural Network Architecture", font=("bold")).pack(anchor="w", pady=(0,5))
        self._add_param_row(main_f, "Hidden Layers:", self.param_vars["mod_num_layers"])
        self._add_param_row(main_f, "Neurons per Layer:", self.param_vars["mod_hidden_dim"])
        
        f_act = ttk.Frame(main_f)
        f_act.pack(fill="x", pady=2)
        ttk.Label(f_act, text="Activation:", width=18).pack(side="left")
        acts = ["tanh", "sigmoid", "relu", "swish", "elu"]
        ttk.Combobox(f_act, textvariable=self.param_vars["mod_activation"], values=acts, state="readonly").pack(side="right", expand=True, fill="x")
        
        ttk.Separator(main_f).pack(fill="x", pady=15)
        
        # 2. Training
        ttk.Label(main_f, text="Training Optimizer", font=("bold")).pack(anchor="w", pady=(0,5))
        self._add_param_row(main_f, "Epochs:", self.param_vars["EPOCHS"])
        self._add_param_row(main_f, "Learning Rate:", self.param_vars["LEARNING_RATE"])
        
        ttk.Separator(main_f).pack(fill="x", pady=15)
        
        # 3. Physics Constants
        ttk.Label(main_f, text="Physics Constants", font=("bold")).pack(anchor="w", pady=(0,5))
        has_consts = False
        for key, var in self.param_vars.items():
            if key.startswith("phys_") and not key.startswith("phys_IC_"):
                label_txt = key.replace("phys_", "") + ":"
                self._add_param_row(main_f, label_txt, var)
                has_consts = True
        if not has_consts: ttk.Label(main_f, text="(None)", foreground="gray").pack(anchor="w")

        ttk.Separator(main_f).pack(fill="x", pady=15)

        # 4. [NUEVO] Initial Conditions
        ttk.Label(main_f, text="Initial Conditions", font=("bold")).pack(anchor="w", pady=(0,5))
        has_ics = False
        for key, var in self.param_vars.items():
            if key.startswith("phys_IC_"):
                label_txt = key.replace("phys_IC_", "") + ":"
                self._add_param_row(main_f, label_txt, var)
                has_ics = True
        if not has_ics: ttk.Label(main_f, text="(Defined by distribution)", foreground="gray").pack(anchor="w")
        
        ttk.Button(main_f, text="Close & Apply", command=win.destroy).pack(side="bottom", pady=10)

    def _add_param_row(self, parent, label_text, variable):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label_text, width=18).pack(side="left")
        ttk.Entry(f, textvariable=variable).pack(side="right", expand=True, fill="x")

    def _on_problem_change(self, event):
        prob = self.prob_combo.get()
        self.shared_state['problem_name'].set(prob)
        self._init_default_parameters(prob)
        self._reset_plots()

    def _toggle_mode(self):
        current_mode = self.shared_state['use_csv_mode'].get()
        self.shared_state['use_csv_mode'].set(not current_mode)
        self._refresh_ui_state()

    def _refresh_ui_state(self):
        use_csv = self.shared_state['use_csv_mode'].get()
        df = self.shared_state.get('current_dataframe')
        if use_csv:
            self.btn_switch_mode.config(text="Switch back to Analytical")
            self.lbl_status_mode.config(text="Current: CSV Data-Driven", foreground="green")
            self.btn_map.config(state="normal")
            if df is not None:
                self.lbl_csv_check.config(text="CSV Loaded: [Yes]", foreground="green")
            else:
                self.lbl_csv_check.config(text="CSV Loaded: [No]", foreground="red")
        else:
            self.btn_switch_mode.config(text="Switch to CSV Mode")
            self.lbl_status_mode.config(text="Current: Analytical", foreground="blue")
            self.btn_map.config(state="disabled")
            self.lbl_csv_check.config(text="CSV Loaded: [Ignored]", foreground="gray")

    def _open_mapping_dialog(self):
        df = self.shared_state.get('current_dataframe')
        if df is None:
            messagebox.showerror("Error", "No CSV loaded! Go to Data Exploration.")
            return
        prob = self.prob_combo.get()
        req = get_problem_variables(prob)
        win = tk.Toplevel(self)
        win.title(f"Map Columns ({prob})")
        win.geometry("350x300")
        combos = {}
        for var in req:
            f = ttk.Frame(win); f.pack(fill="x", padx=10, pady=5)
            ttk.Label(f, text=f"{var}:").pack(side="left")
            cb = ttk.Combobox(f, values=list(df.columns), state="readonly")
            cb.pack(side="right")
            combos[var] = cb
        def save():
            mapping = {v: c.get() for v, c in combos.items()}
            if all(mapping.values()):
                self.shared_state['column_mapping'] = mapping
                messagebox.showinfo("Saved", "Mapping Saved!")
                win.destroy()
            else: messagebox.showerror("Error", "Map all fields.")
        ttk.Button(win, text="Save Mapping", command=save).pack(pady=15)

    # --- TRAINING LOGIC ---
    def toggle_training(self):
        if not self.shared_state['is_training']: self.start_training()
        else: self.stop_training()

    def start_training(self):
        use_csv = self.shared_state['use_csv_mode'].get()
        df = self.shared_state.get('current_dataframe')
        mapping = self.shared_state.get('column_mapping')
        if use_csv:
            if df is None: messagebox.showerror("Error", "No CSV loaded."); return
            if mapping is None: messagebox.showerror("Error", "Columns not mapped."); return

        self.shared_state['is_training'] = True
        self.btn_train.config(text="⏹ STOP TRAINING")
        
        prob = self.prob_combo.get()
        conf = get_active_config(prob)
        
        # --- INYECTAR PARAMETROS DESDE GUI ---
        conf["EPOCHS"] = self.param_vars["EPOCHS"].get()
        conf["LEARNING_RATE"] = self.param_vars["LEARNING_RATE"].get()
        if "MODEL_CONFIG" not in conf: conf["MODEL_CONFIG"] = {}
        conf["MODEL_CONFIG"]["num_layers"] = self.param_vars["mod_num_layers"].get()
        conf["MODEL_CONFIG"]["hidden_dim"] = self.param_vars["mod_hidden_dim"].get()
        conf["MODEL_CONFIG"]["activation"] = self.param_vars["mod_activation"].get()
        
        # Inyectar Fisica y Condiciones Iniciales
        for k, v in self.param_vars.items():
            if k.startswith("phys_IC_"):
                # Es una condicion inicial (Nested dict)
                ic_key = k.replace("phys_IC_", "")
                if "initial_conditions" not in conf["PHYSICS_CONFIG"]:
                    conf["PHYSICS_CONFIG"]["initial_conditions"] = {}
                conf["PHYSICS_CONFIG"]["initial_conditions"][ic_key] = v.get()
            elif k.startswith("phys_"):
                # Es una constante fisica normal
                conf["PHYSICS_CONFIG"][k.replace("phys_", "")] = v.get()

        self.trainer = PINNTrainer(conf, prob, csv_data=(df if use_csv else None), column_mapping=(mapping if use_csv else None))
        self.shared_state['trainer'] = self.trainer
        threading.Thread(target=self._loop, daemon=True).start()

    def stop_training(self):
        self.shared_state['is_training'] = False
        self.btn_train.config(text="▶ START TRAINING")

    def _loop(self):
        try:
            while self.shared_state['is_training']:
                losses = self.trainer.perform_one_step()
                ep = self.trainer.epoch
                if ep % 10 == 0: self.after(0, self._update_gui, ep, losses[0])
                if ep >= self.trainer.config["EPOCHS"]: self.stop_training(); break
        except Exception as e:
            print(f"Error: {e}")
            self.stop_training()

    def _update_gui(self, ep, loss):
        self.lbl_epoch.config(text=f"Epoch: {ep}")
        self.lbl_loss.config(text=f"Loss: {loss:.2e}")
        
        self.ax_loss.clear()
        hist = self.trainer.loss_history
        plot_hist = hist if len(hist) < 2000 else hist[-2000:]
        self.ax_loss.plot(plot_hist, label='Loss', color='#1f77b4')
        self.ax_loss.set_yscale('log')
        self.ax_loss.set_title("Training Convergence")
        self.ax_loss.grid(True, alpha=0.3)

        try:
            phys = self.trainer.physics
            model = self.trainer.model
            prob = self.trainer.active_problem
            config = self.trainer.config["PHYSICS_CONFIG"]

            if prob in ["SHO", "DHO"]:
                self.ax_hm_pred.axis('off'); self.ax_hm_true.axis('off')
                self.ax_slice.axis('on'); self.ax_slice.clear()

                if phys.csv_data is not None and not phys.has_analytical:
                    t_col = self.trainer.column_mapping['time']
                    t_min, t_max = phys.csv_data[t_col].min(), phys.csv_data[t_col].max()
                else:
                    t_min, t_max = phys.domain_config["t_domain"]
                
                t_grid = np.linspace(t_min, t_max, 100).reshape(-1, 1).astype(np.float32)
                t_tf = tf.convert_to_tensor(t_grid)
                
                if phys.has_analytical:
                    u_true = phys.analytical_solution(t_tf).numpy()
                    self.ax_slice.plot(t_grid, u_true, 'k--', label='Analytical')
                elif phys.csv_data is not None:
                    t_c = self.trainer.column_mapping['time']
                    x_c = self.trainer.column_mapping['displacement']
                    self.ax_slice.scatter(phys.csv_data[t_c], phys.csv_data[x_c], s=10, c='gray', alpha=0.5, label='Data')

                u_pred = model(t_tf).numpy()
                self.ax_slice.plot(t_grid, u_pred, 'r-', label='PINN Prediction')
                self.ax_slice.legend(fontsize='x-small')
                self.ax_slice.set_title(f"Dynamics ({prob})")
                self.ax_slice.grid(True)

            elif prob == "HEAT":
                self.ax_hm_pred.axis('on'); self.ax_hm_true.axis('on')
                self.ax_slice.axis('on'); self.ax_slice.clear(); self.ax_hm_pred.clear(); self.ax_hm_true.clear()

                x_min, x_max = config["x_domain"]
                y_min, y_max = config["y_domain"]
                t_mid = sum(config["t_domain"]) / 2
                y_mid = sum(config["y_domain"]) / 2

                x_grid = np.linspace(x_min, x_max, 100).reshape(-1, 1).astype(np.float32)
                y_ones = np.full_like(x_grid, y_mid); t_ones = np.full_like(x_grid, t_mid)
                xyt_slice = np.hstack([x_grid, y_ones, t_ones])
                
                u_slice_pred = model(tf.convert_to_tensor(xyt_slice)).numpy()
                u_slice_true = phys.analytical_solution(tf.convert_to_tensor(xyt_slice)).numpy()

                self.ax_slice.plot(x_grid, u_slice_true, 'k--', label='Analytical')
                self.ax_slice.plot(x_grid, u_slice_pred, 'r-', label='PINN')
                self.ax_slice.set_title(f"Slice y={y_mid:.1f}, t={t_mid:.2f}")
                self.ax_slice.legend(fontsize='x-small')
                self.ax_slice.grid(True)

                # Heatmaps
                N = 40
                x = np.linspace(x_min, x_max, N); y = np.linspace(y_min, y_max, N)
                X, Y = np.meshgrid(x, y); T = np.full_like(X, t_mid)
                inp = np.hstack([X.flatten().reshape(-1,1), Y.flatten().reshape(-1,1), T.flatten().reshape(-1,1)]).astype(np.float32)
                
                Z_pred = model(tf.convert_to_tensor(inp)).numpy().reshape(N, N)
                Z_true = phys.analytical_solution(tf.convert_to_tensor(inp)).numpy().reshape(N, N)
                
                vmin = min(Z_true.min(), Z_pred.min()); vmax = max(Z_true.max(), Z_pred.max())
                self.ax_hm_pred.imshow(Z_pred, extent=[x_min,x_max,y_min,y_max], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                self.ax_hm_pred.set_title("PINN")
                self.ax_hm_true.imshow(Z_true, extent=[x_min,x_max,y_min,y_max], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                self.ax_hm_true.set_title("Analytical")

        except Exception as e:
            print(f"Plot Error: {e}")

        self.canvas.draw()