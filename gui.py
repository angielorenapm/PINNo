# gui.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import pandas as pd

from src.config import get_active_config
from src.training import PINNTrainer


class PINNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PINN Interactive Trainer")
        self.root.geometry("1200x700")

        # ===== Estado general =====
        self.trainer = None
        self.is_training = False
        self.epoch = 0
        self.loss_history = []
        self.problem_name = tk.StringVar(value="SHO")

        # ===== Estado Data Exploration =====
        self.df = None
        self.y_choice = tk.StringVar(value="x (m)")  # opción por defecto

        # ===== Notebook con 3 pestañas =====
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Pestaña 1: Data Exploration ---
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data Exploration")
        self._build_data_tab()

        # --- Pestaña 2: Train & Visualize ---
        self.train_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_tab, text="Train & Visualize")
        self._build_train_tab()

        # --- Pestaña 3: Metrics Report ---
        self.report_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.report_tab, text="Metrics Report")
        self._build_report_tab()

    # =========================================================
    # TAB 1 - DATA EXPLORATION (CSV con nombres EXACTOS)
    # =========================================================
    def _build_data_tab(self):
        top = ttk.Frame(self.data_tab, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open File...", command=self._open_csv).pack(side=tk.LEFT, padx=5)

        ttk.Label(top, text="Y variable:").pack(side=tk.LEFT, padx=(15, 5))
        self.y_selector = ttk.Combobox(
            top,
            textvariable=self.y_choice,
            values=["x (m)", "v (m/s)", "a (m/s^2)"],
            state="readonly",
            width=15
        )
        self.y_selector.pack(side=tk.LEFT)
        ttk.Button(top, text="Plot", command=self._plot_csv).pack(side=tk.LEFT, padx=10)

        main = ttk.Frame(self.data_tab, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Lista de atributos
        left = ttk.Labelframe(main, text="Attributes", padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        columns = ("#", "Attribute")
        self.attr_tree = ttk.Treeview(left, columns=columns, show="headings", height=16)
        self.attr_tree.heading("#", text="#")
        self.attr_tree.heading("Attribute", text="Attribute")
        self.attr_tree.column("#", width=40, anchor="center")
        self.attr_tree.column("Attribute", width=200, anchor="w")
        self.attr_tree.pack(fill=tk.BOTH, expand=True)

        # ---- Contenedor derecho con dos gráficas lado a lado ----
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plots_row = ttk.Frame(right)
        plots_row.pack(fill=tk.BOTH, expand=True)

        # Serie temporal (Y vs t (s))
        self.ts_frame = ttk.Frame(plots_row)
        self.ts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.ts_fig = plt.Figure(figsize=(6.0, 4.6), dpi=100)
        self.ts_ax = self.ts_fig.add_subplot(111)
        self.ts_ax.set_title("Time series")
        self.ts_ax.set_xlabel("t (s)")
        self.ts_ax.set_ylabel("Value")
        self.ts_ax.grid(True, linestyle="--", linewidth=0.5)
        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, master=self.ts_frame)
        self.ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Matriz de correlación (x, v, a)
        self.corr_frame = ttk.Frame(plots_row)
        self.corr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.corr_fig = plt.Figure(figsize=(6.0, 4.6), dpi=100)
        self.corr_ax = self.corr_fig.add_subplot(111)
        self.corr_ax.set_title("Correlation matrix")
        self.corr_canvas = FigureCanvasTkAgg(self.corr_fig, master=self.corr_frame)
        self.corr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        info = ttk.Labelframe(right, text="Dataset Info", padding=10)
        info.pack(fill=tk.X, pady=6)
        self.info_label = ttk.Label(info, text="No file loaded")
        self.info_label.pack(anchor="w")

    def _open_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.df = pd.read_csv(path)

            # Validación estricta de nombres
            if "t (s)" not in self.df.columns:
                messagebox.showerror("Missing column", "CSV must contain the column 't (s)'.")
                self.df = None
                return

            present_targets = [c for c in ["x (m)", "v (m/s)", "a (m/s^2)"] if c in self.df.columns]
            if not present_targets:
                messagebox.showerror(
                    "Invalid file",
                    "CSV must contain at least one of: 'x (m)', 'v (m/s)', 'a (m/s^2)'."
                )
                self.df = None
                return

            self.y_selector.config(values=present_targets)
            if self.y_choice.get() not in present_targets:
                self.y_choice.set(present_targets[0])

            # Llenar lista de atributos
            self.attr_tree.delete(*self.attr_tree.get_children())
            for i, col in enumerate(self.df.columns, 1):
                self.attr_tree.insert("", "end", values=(i, str(col)))

            self.info_label.config(text=f"File loaded: {path.split('/')[-1]}")
            self._plot_csv()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file:\n\n{e}")

    def _plot_csv(self):
        if self.df is None:
            messagebox.showinfo("No file", "Please open a CSV file first.")
            return

        time_col = "t (s)"
        y_col = self.y_choice.get()
        if y_col not in self.df.columns:
            messagebox.showerror("Missing column", f"The selected column '{y_col}' is not in the CSV file.")
            return

        # Serie temporal
        self.ts_ax.clear()
        self.ts_ax.plot(self.df[time_col], self.df[y_col], label=y_col)
        self.ts_ax.set_title(f"{y_col} vs t (s)")
        self.ts_ax.set_xlabel("t (s)")
        self.ts_ax.set_ylabel(y_col)
        self.ts_ax.grid(True, linestyle="--", linewidth=0.5)
        self.ts_ax.legend()
        self.ts_fig.tight_layout()
        self.ts_canvas.draw()

        # Matriz de correlación (solo con columnas presentes)
        cols_for_corr = [c for c in ["x (m)", "v (m/s)", "a (m/s^2)"] if c in self.df.columns]
        self.corr_ax.clear()
        if len(cols_for_corr) >= 2:
            corr = self.df[cols_for_corr].corr(numeric_only=True)
            im = self.corr_ax.imshow(corr.values, interpolation="nearest")
            self.corr_ax.set_xticks(range(len(cols_for_corr)))
            self.corr_ax.set_yticks(range(len(cols_for_corr)))
            self.corr_ax.set_xticklabels(cols_for_corr, rotation=45, ha="right")
            self.corr_ax.set_yticklabels(cols_for_corr)
            self.corr_ax.set_title("Correlation matrix")
            for i in range(len(cols_for_corr)):
                for j in range(len(cols_for_corr)):
                    self.corr_ax.text(j, i, f"{corr.values[i, j]:.2f}",
                                      ha="center", va="center", fontsize=9)
            self.corr_fig.colorbar(im, ax=self.corr_ax, fraction=0.046, pad=0.04)
        else:
            self.corr_ax.set_title("Correlation matrix (need ≥ 2 vars)")
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()

    # =========================================================
    # TAB 2 - TRAIN & VISUALIZE (tu GUI original + panel métricas)
    # =========================================================
    def _build_train_tab(self):
        main_frame = ttk.Frame(self.train_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Labelframe(main_frame, text="Controles", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        right_col = ttk.Frame(main_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Área de plots (igual que tu GUI original) ---
        self.plot_frame = ttk.Frame(right_col)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(control_frame, text="Seleccionar Problema:").pack(pady=(0, 5), anchor="w")
        self.problem_selector = ttk.Combobox(
            control_frame, textvariable=self.problem_name,
            values=["SHO", "DHO", "WAVE"], state="readonly"
        )
        self.problem_selector.pack(pady=5, fill=tk.X)

        self.start_button = ttk.Button(control_frame, text="Iniciar Entrenamiento", command=self.start_training)
        self.start_button.pack(pady=10, fill=tk.X)

        self.stop_button = ttk.Button(control_frame, text="Detener Entrenamiento",
                                      command=self.stop_training, state=tk.DISABLED)
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

        # --- Panel extra: resumen de métricas estilo "summary" ---
        summary_box = ttk.Labelframe(right_col, text="Cross-validation style summary", padding=8)
        summary_box.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=(8, 0))
        self.metrics_text = tk.Text(summary_box, height=10, width=80)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.configure(state="normal")
        self.metrics_text.insert(tk.END, "Run training to see the live summary here...\n")
        self.metrics_text.configure(state="disabled")

    # =========================================================
    # TAB 3 - METRICS REPORT (texto + gráfica/imagen)
    # =========================================================
    def _build_report_tab(self):
        top = ttk.Frame(self.report_tab, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Open Report Image...", command=self._open_report_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Save Report (.txt)", command=self._save_report_txt).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Save Figure (.png)", command=self._save_report_png).pack(side=tk.LEFT, padx=5)

        body = ttk.Frame(self.report_tab, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        # Texto del informe (renombrado a "Report")
        left = ttk.Labelframe(body, text="Report", padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.report_text = tk.Text(left, height=25)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self.report_text.configure(state="normal")
        self.report_text.insert(tk.END, "Run training to populate the report here...\n")
        self.report_text.configure(state="disabled")

        # Panel derecho: aquí mostraremos la misma gráfica Pred vs Analítica
        right = ttk.Labelframe(body, text="Report Plot", padding=8)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.report_fig = plt.Figure(figsize=(6.0, 4.6), dpi=100)
        self.report_ax = self.report_fig.add_subplot(111)
        self.report_ax.set_title("No plot yet")
        self.report_ax.grid(True, linestyle="--", linewidth=0.5)
        self.report_canvas = FigureCanvasTkAgg(self.report_fig, master=right)
        self.report_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _open_report_image(self):
        """Permite cargar una imagen externa en el panel derecho.
        En cuanto vuelva a actualizarse el entrenamiento, la gráfica
        Pred vs Analítica reemplazará esta imagen automáticamente."""
        path = filedialog.askopenfilename(
            title="Select report image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = plt.imread(path)
            self.report_ax.clear()
            self.report_ax.imshow(img)
            self.report_ax.axis("off")
            self.report_fig.tight_layout()
            self.report_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image:\n\n{e}")

    def _save_report_txt(self):
        path = filedialog.asksaveasfilename(
            title="Save report as",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")]
        )
        if not path:
            return
        try:
            content = self.report_text.get("1.0", tk.END)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content.strip() + "\n")
            messagebox.showinfo("Saved", f"Report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save report:\n\n{e}")

    def _save_report_png(self):
        path = filedialog.asksaveasfilename(
            title="Save figure as",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return
        try:
            self.report_fig.savefig(path, dpi=150, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Figure saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save figure:\n\n{e}")

    # ======== LÓGICA DE ENTRENAMIENTO ========
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
            self.trainer = PINNTrainer(active_config, problem)
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
        if not self.is_training:
            return
        try:
            losses = self.trainer.perform_one_step()
            total_loss = losses[0].numpy()
            self.loss_history.append(total_loss)
            self.epoch += 1

            if self.epoch % 10 == 0 or self.epoch == 1:
                self.epoch_label.config(text=f"Época: {self.epoch}")
                self.loss_label.config(text=f"Pérdida (Loss): {total_loss:.4e}")

            if self.epoch == 1 or self.epoch % 100 == 0:
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

            # Actualiza resumen + duplica gráfica en pestaña 3
            self._update_metrics_panel(x_true.flatten(), x_pred.flatten())
            self._update_report_graph(t_plot, x_true, x_pred)

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

            self._update_metrics_panel(u_true.flatten(), u_pred.flatten())
            self._update_report_graph(x_plot, u_true, u_pred)

        self.ax_solution.relim()
        self.ax_solution.autoscale_view()
        self.canvas.draw()

    # --- NUEVO: gráfica Pred vs Analítica en pestaña "Metrics Report" ---
    def _update_report_graph(self, x, y_true, y_pred):
        try:
            self.report_ax.clear()
            self.report_ax.plot(x, y_true, 'r--', label="Analítica")
            self.report_ax.plot(x, y_pred, 'b-', label="Predicha")
            self.report_ax.set_title("Solución Predicha vs Analítica")
            self.report_ax.set_xlabel("Dominio")
            self.report_ax.set_ylabel("Magnitud")
            self.report_ax.grid(True, linestyle="--", linewidth=0.5)
            self.report_ax.legend()
            self.report_fig.tight_layout()
            self.report_canvas.draw()
        except Exception as e:
            print(f"[WARN] No se pudo actualizar la gráfica del reporte: {e}")

    # ------- Métricas estilo WEKA (regresión + confusión binaria) -------
    def _update_metrics_panel(self, y_true, y_pred):
        n = len(y_true)
        if n == 0:
            return

        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))

        # Errores relativos tipo Weka
        y_bar = float(np.mean(y_true))
        rae = float(np.sum(np.abs(err)) / (np.sum(np.abs(y_true - y_bar)) + 1e-12)) * 100.0
        rrse = float(np.sqrt(np.sum(err**2) / (np.sum((y_true - y_bar)**2) + 1e-12))) * 100.0

        # Clasificación binaria por mediana (para matriz de confusión visual)
        thr = float(np.median(y_true))
        true_cls = (y_true >= thr).astype(int)
        pred_cls = (y_pred >= thr).astype(int)

        tp = int(np.sum((true_cls == 1) & (pred_cls == 1)))
        tn = int(np.sum((true_cls == 0) & (pred_cls == 0)))
        fp = int(np.sum((true_cls == 0) & (pred_cls == 1)))
        fn = int(np.sum((true_cls == 1) & (pred_cls == 0)))

        acc = (tp + tn) / n if n else 0.0

        def prf(tp_, fp_, fn_):
            prec = tp_ / (tp_ + fp_ + 1e-12)
            rec = tp_ / (tp_ + fn_ + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            return prec, rec, f1

        prec_a, rec_a, f1_a = prf(tn, fn, fp)  # clase 0
        prec_b, rec_b, f1_b = prf(tp, fp, fn)  # clase 1

        denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-12
        mcc = ((tp*tn - fp*fn) / denom)

        lines = []
        lines.append("=== Summary ===")
        lines.append(f"Correctly Classified Instances    {int(round(acc*n))}    {acc*100:6.4f} %")
        lines.append(f"Incorrectly Classified Instances  {n-int(round(acc*n))}    {(1-acc)*100:6.4f} %")
        lines.append(f"Mean absolute error               {mae:0.6f}")
        lines.append(f"Root mean squared error           {rmse:0.6f}")
        lines.append(f"Relative absolute error           {rae:0.4f} %")
        lines.append(f"Root relative squared error       {rrse:0.4f} %")
        lines.append(f"Total Number of Instances         {n}")
        lines.append("")
        lines.append("=== Detailed Accuracy By Class ===")
        lines.append("              Precision   Recall     F1-Score     MCC")
        lines.append(f"a = LOW        {prec_a:0.3f}      {rec_a:0.3f}     {f1_a:0.3f}     {mcc:0.3f}")
        lines.append(f"b = HIGH       {prec_b:0.3f}      {rec_b:0.3f}     {f1_b:0.3f}     {mcc:0.3f}")
        lines.append("")
        lines.append("=== Confusion Matrix ===")
        lines.append("  a  b  <-- classified as")
        lines.append(f"{tn:3d} {fp:2d} | a = LOW")
        lines.append(f"{fn:3d} {tp:2d} | b = HIGH")

        report = "\n".join(lines)

        # Actualiza panel de métricas de la pestaña 2
        if hasattr(self, "metrics_text"):
            self.metrics_text.configure(state="normal")
            self.metrics_text.delete("1.0", tk.END)
            self.metrics_text.insert(tk.END, report)
            self.metrics_text.configure(state="disabled")

        # Actualiza el texto del informe en la pestaña 3
        if hasattr(self, "report_text"):
            self.report_text.configure(state="normal")
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, report)
            self.report_text.configure(state="disabled")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    root = tk.Tk()
    app = PINNGUI(root)
    root.mainloop()
