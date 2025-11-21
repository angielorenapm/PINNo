# pinno/gui_modules/data_explorer.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import csv

from .components import PlotManager, DataLoader

class DataExplorer(ttk.Frame):
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.data_loader = DataLoader()
        self.plot_manager = PlotManager()
        self.y_choice = tk.StringVar()
        
        self._build_interface()

    def _build_interface(self):
        # --- Panel Superior ---
        control_frame = ttk.Frame(self, padding=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(control_frame, text="📂 Open CSV...", command=self._handle_file_open).pack(side=tk.LEFT, padx=5)
        self.lbl_filename = ttk.Label(control_frame, text="[No Data]", foreground="gray")
        self.lbl_filename.pack(side=tk.LEFT, padx=5)

        # Selector Y (Filtrado)
        ttk.Label(control_frame, text="| Plot Variable Y:").pack(side=tk.LEFT, padx=(20, 5))
        self.y_selector = ttk.Combobox(control_frame, textvariable=self.y_choice, state="readonly", width=15)
        self.y_selector.pack(side=tk.LEFT)
        self.y_selector.bind('<<ComboboxSelected>>', self._on_variable_change)
        
        self.lbl_time_ax = ttk.Label(control_frame, text="X-Axis: [Index]", foreground="blue")
        self.lbl_time_ax.pack(side=tk.LEFT, padx=15)

        # --- Layout Principal ---
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 1. Izquierda: Atributos
        left_frame = ttk.Frame(main_pane, width=220)
        main_pane.add(left_frame, weight=0)
        
        attr_group = ttk.LabelFrame(left_frame, text="Attributes", padding=5)
        attr_group.pack(fill="both", expand=True)
        
        self.attr_tree = ttk.Treeview(attr_group, columns=("#", "Name"), show="headings", height=20)
        self.attr_tree.heading("#", text="#")
        self.attr_tree.heading("Name", text="Name")
        self.attr_tree.column("#", width=30, anchor="center")
        self.attr_tree.column("Name", width=140, anchor="w")
        
        sb = ttk.Scrollbar(attr_group, orient="vertical", command=self.attr_tree.yview)
        self.attr_tree.configure(yscrollcommand=sb.set)
        self.attr_tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # 2. Derecha: Graficas
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=4)

        plots_area = ttk.Frame(right_frame)
        plots_area.pack(fill="both", expand=True)
        
        self.ts_frame = ttk.Frame(plots_area)
        self.ts_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 5))
        
        self.corr_frame = ttk.Frame(plots_area)
        self.corr_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(5, 0))

        self.ts_fig, self.ts_ax, self.ts_canvas = self.plot_manager.create_time_series_plot(self.ts_frame)
        self.corr_fig, self.corr_ax, self.corr_canvas = self.plot_manager.create_correlation_plot(self.corr_frame)

        # 3. Inferior: Stats
        stats_frame = ttk.LabelFrame(right_frame, text="Dataset Info", padding=5)
        stats_frame.pack(side="bottom", fill="x", pady=(10, 0))
        self.txt_stats = tk.Text(stats_frame, height=6, font=("Consolas", 8), bg="#f4f4f4")
        self.txt_stats.pack(fill="both")

    def _handle_file_open(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path: return
        
        try:
            has_header = self._detect_header(path)
            header_opt = 0 if has_header else None
            df = pd.read_csv(path, header=header_opt)
            if header_opt is None: df.columns = [f"col_{i}" for i in range(df.shape[1])]
            
            self.shared_state['current_dataframe'] = df
            self.shared_state['external_data_path'] = path
            self.lbl_filename.config(text=path.split("/")[-1])
            
            self._update_attributes(df)
            self._prompt_time_column(df) # AQUI se configura el filtro Y
            
            self._update_stats(df)
            self._plot_correlation(df)
            
        except Exception as e:
            messagebox.showerror("Error", f"Load failed:\n{e}")

    def _detect_header(self, path):
        try:
            with open(path, 'r') as f: return csv.Sniffer().has_header(f.read(1024))
        except: return True

    def _prompt_time_column(self, df):
        win = tk.Toplevel(self)
        win.title("Select Time Column")
        win.geometry("300x150")
        win.transient(self)
        win.grab_set()
        
        ttk.Label(win, text="Which column is TIME (t)?").pack(pady=10)
        cb = ttk.Combobox(win, values=list(df.columns), state="readonly")
        cb.pack()
        if len(df.columns)>0: cb.current(0)
        
        def confirm():
            col = cb.get()
            self.shared_state['time_col'] = col
            self.lbl_time_ax.config(text=f"X-Axis: {col}")
            
            # --- CAMBIO: FILTRAR EJE Y ---
            # Todas las columnas EXCEPTO la de tiempo
            y_options = [c for c in df.columns if c != col]
            self.y_selector['values'] = y_options
            if y_options:
                self.y_selector.current(0)
                self._plot_ts()
            
            win.destroy()
            
        ttk.Button(win, text="Confirm", command=confirm).pack(pady=10)
        self.wait_window(win)

    def _update_attributes(self, df):
        for i in self.attr_tree.get_children(): self.attr_tree.delete(i)
        for idx, col in enumerate(df.columns):
            self.attr_tree.insert("", "end", values=(idx, col))

    def _update_stats(self, df):
        self.txt_stats.config(state="normal")
        self.txt_stats.delete("1.0", tk.END)
        try: self.txt_stats.insert("1.0", df.describe().to_string())
        except: self.txt_stats.insert("1.0", "No numeric stats.")
        self.txt_stats.config(state="disabled")

    def _plot_correlation(self, df):
        self.plot_manager.plot_correlation_matrix(df, self.corr_ax, self.corr_canvas)

    def _on_variable_change(self, e):
        self._plot_ts()

    def _plot_ts(self):
        df = self.shared_state.get('current_dataframe')
        y = self.y_choice.get()
        t = self.shared_state.get('time_col')
        
        if df is not None and y:
            self.ts_ax.clear()
            if t and t in df.columns:
                tmp = df.sort_values(t)
                self.ts_ax.plot(tmp[t], tmp[y], label=y)
                self.ts_ax.set_xlabel(t)
            else:
                self.ts_ax.plot(df[y], label=y)
                self.ts_ax.set_xlabel("Index")
            
            self.ts_ax.set_ylabel(y)
            self.ts_ax.set_title("Time Series")
            self.ts_ax.legend()
            self.ts_ax.grid(True, alpha=0.3)
            self.ts_canvas.draw()