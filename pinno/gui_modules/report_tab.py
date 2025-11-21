# pinno/gui_modules/report_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import json
import tensorflow as tf

from .components import MetricsCalculator

class ReportTab(ttk.Frame):
    """
    Pestaña de Reportes Avanzada:
    - Izquierda: Estadísticas detalladas (Texto).
    - Derecha: Doble gráfico (Loss + Solución) para un reporte completo.
    """
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.training_tab_ref = None
        self.metrics_calc = MetricsCalculator()
        
        self._setup_ui()

    def _setup_ui(self):
        # --- TOOLBAR (Botones) ---
        toolbar = ttk.Frame(self, padding=5)
        toolbar.pack(side="top", fill="x")
        
        ttk.Label(toolbar, text="Actions: ").pack(side="left", padx=(0, 10))
        
        self.btn_gen = ttk.Button(toolbar, text="🔄 Generate Analysis", command=self.generate_report)
        self.btn_gen.pack(side="left", padx=2)
        
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10, pady=2)
        
        self.btn_save_txt = ttk.Button(toolbar, text="Save Report (.txt)", command=self.save_report_txt, state="disabled")
        self.btn_save_txt.pack(side="left", padx=2)
        
        self.btn_save_fig = ttk.Button(toolbar, text="Save Figure (.png)", command=self.save_figure, state="disabled")
        self.btn_save_fig.pack(side="left", padx=2)
        
        self.btn_save_csv = ttk.Button(toolbar, text="Save Data (.csv)", command=self.save_data, state="disabled")
        self.btn_save_csv.pack(side="left", padx=2)
        
        self.btn_save_model = ttk.Button(toolbar, text="Save Model (.keras)", command=self.save_model, state="disabled")
        self.btn_save_model.pack(side="left", padx=2)

        # --- SPLIT VIEW (Texto | Gráficas) ---
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=10, pady=5)

        # === IZQUIERDA: TEXTO ===
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="=== Analysis Report ===", font=("Consolas", 10, "bold")).pack(anchor="w")
        
        text_scroll = ttk.Scrollbar(left_frame)
        text_scroll.pack(side="right", fill="y")
        self.txt_report = tk.Text(left_frame, height=20, width=50, font=("Consolas", 9), 
                                 yscrollcommand=text_scroll.set, relief="sunken", bd=1)
        self.txt_report.pack(fill="both", expand=True)
        text_scroll.config(command=self.txt_report.yview)
        
        conf_frame = ttk.LabelFrame(left_frame, text="Config Used", padding=5)
        conf_frame.pack(fill="x", pady=(10, 0))
        self.txt_config = tk.Text(conf_frame, height=6, font=("Consolas", 8), bg="#f0f0f0")
        self.txt_config.pack(fill="both")

        # === DERECHA: VISUALIZACION DOBLE ===
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        ttk.Label(right_frame, text="=== Model Visualization ===", font=("Helvetica", 10, "bold")).pack(anchor="w")
        
        # Crear Figura con 2 Subplots (Arriba: Loss, Abajo: Solución)
        self.fig, (self.ax_loss, self.ax_sol) = plt.subplots(2, 1, figsize=(6, 8), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.fig.subplots_adjust(hspace=0.3) # Espacio entre gráficas
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._reset_view()

    def _reset_view(self):
        self.txt_report.delete("1.0", tk.END)
        self.txt_report.insert("1.0", "Train a model and press 'Generate Analysis'.")
        self.txt_config.delete("1.0", tk.END)
        
        for ax in [self.ax_loss, self.ax_sol]:
            ax.clear()
            ax.axis('off')
            ax.text(0.5, 0.5, "No Data", ha='center')
        
        self.canvas.draw()

    def generate_report(self):
        trainer = self.shared_state.get('trainer')
        if not trainer or not trainer.loss_history:
            messagebox.showwarning("Info", "No training data found.")
            return

        # Habilitar botones
        for btn in [self.btn_save_txt, self.btn_save_fig, self.btn_save_csv, self.btn_save_model]:
            btn.config(state="normal")

        try:
            # 1. Datos
            y_true, y_pred, t_grid = self._get_evaluation_data(trainer)
            hist = trainer.loss_history

            # 2. Reporte Texto
            report_str = self.metrics_calc.comprehensive_report(y_true, y_pred)
            self.txt_report.config(state="normal")
            self.txt_report.delete("1.0", tk.END)
            self.txt_report.insert("1.0", report_str)
            
            self._show_config(trainer.config)

            # 3. GRAFICA 1: LOSS (Arriba)
            self.ax_loss.clear()
            self.ax_loss.axis('on')
            self.ax_loss.plot(hist, label='Training Loss', color='#e74c3c')
            self.ax_loss.set_yscale('log')
            self.ax_loss.set_title("1. Training Convergence (Loss)")
            self.ax_loss.set_ylabel("Loss (Log)")
            self.ax_loss.grid(True, alpha=0.3)
            self.ax_loss.legend()

            # 4. GRAFICA 2: SOLUCION (Abajo)
            self._plot_analysis(t_grid, y_true, y_pred, trainer)

            self.canvas.draw()

        except Exception as e:
            print(f"Report Error: {e}")
            messagebox.showerror("Error", f"Failed: {e}")

    def _plot_analysis(self, t, y_true, y_pred, trainer):
        self.ax_sol.clear()
        self.ax_sol.axis('on')
        
        # Ground Truth
        if len(y_true) > 0:
            if trainer.use_csv:
                self.ax_sol.scatter(t, y_true, label="Data (Ground Truth)", color='gray', s=15, alpha=0.6)
            else:
                self.ax_sol.plot(t, y_true, 'k--', label="Analytical Solution", linewidth=2)
        
        # Prediccion
        self.ax_sol.plot(t, y_pred, 'b-', label="PINN Prediction", linewidth=2)
        
        self.ax_sol.set_title("2. Final Solution Profile")
        self.ax_sol.set_xlabel("Time / Space")
        self.ax_sol.set_ylabel("u")
        self.ax_sol.legend()
        self.ax_sol.grid(True, alpha=0.3)

    def _get_evaluation_data(self, trainer):
        model = trainer.model
        phys = trainer.physics
        prob = trainer.active_problem
        
        # Caso simple (1D)
        if prob in ["SHO", "DHO"]:
            t_d = phys.domain_config["t_domain"]
            # Si es CSV, usamos los puntos reales para calcular error preciso
            if phys.csv_data is not None:
                t_col = trainer.column_mapping['time']
                x_col = trainer.column_mapping['displacement']
                t = phys.csv_data[t_col].values.reshape(-1, 1).astype(np.float32)
                y_true = phys.csv_data[x_col].values.reshape(-1, 1).astype(np.float32)
            else:
                # Si es analitico, generamos grilla densa
                t = np.linspace(t_d[0], t_d[1], 200).reshape(-1, 1).astype(np.float32)
                t_tf = tf.convert_to_tensor(t)
                y_true = phys.analytical_solution(t_tf).numpy()

            y_pred = model(tf.convert_to_tensor(t)).numpy()
            return y_true, y_pred, t
            
        # Caso Complejo (HEAT) - Simplificado para reporte
        return np.array([]), np.array([]), np.array([])

    def _show_config(self, config):
        self.txt_config.config(state="normal")
        self.txt_config.delete("1.0", tk.END)
        s = f"LR: {config.get('LEARNING_RATE')} | Epochs: {config.get('EPOCHS')}\n"
        if 'MODEL_CONFIG' in config:
            mc = config['MODEL_CONFIG']
            s += f"Arch: {mc.get('num_layers')} layers x {mc.get('hidden_dim')} neurons\n"
            s += f"Act: {mc.get('activation')}"
        self.txt_config.insert("1.0", s)
        self.txt_config.config(state="disabled")

    # --- EXPORTACION ---
    def save_figure(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=300)
            messagebox.showinfo("Saved", "Full report figure saved.")

    def save_report_txt(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            with open(path, "w") as f: f.write(self.txt_report.get("1.0", tk.END))
            messagebox.showinfo("Saved", "Report text saved.")

    def save_data(self):
        tr = self.shared_state.get('trainer')
        if not tr: return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            pd.DataFrame(tr.loss_history, columns=["loss"]).to_csv(path)
            messagebox.showinfo("Saved", "Loss data saved.")

    def save_model(self):
        tr = self.shared_state.get('trainer')
        if not tr: return
        path = filedialog.asksaveasfilename(defaultextension=".keras", filetypes=[("Keras", "*.keras")])
        if path:
            tr.model.save(path)
            messagebox.showinfo("Saved", "Model saved.")