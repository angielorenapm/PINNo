# gui_modules/data_explorer.py
"""
Módulo para la pestaña de exploración de datos - Versión corregida
- Matriz de correlación se plotea solo al cargar CSV
- Series temporales se actualizan al cambiar variable Y
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui_modules.components import DataLoader, PlotManager


class DataExplorer(ttk.Frame):
    """Pestaña de exploración y visualización de datos CSV"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.data_loader = DataLoader()
        self.plot_manager = PlotManager()
        
        # Variables de UI
        self.y_choice = tk.StringVar(value="Select variable")
        self._is_plotting = False  # Protección contra múltiples plots
        
        self._build_interface()
        self._setup_event_handlers()

    def _build_interface(self):
        """Construir la interfaz de la pestaña"""
        # Panel de controles
        self._build_control_panel()
        
        # Panel principal
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Lista de atributos
        self._build_attributes_panel(main_frame)
        
        # Área de gráficas
        self._build_plots_panel(main_frame)

    def _build_control_panel(self):
        """Panel superior con controles"""
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(control_frame, text="Open CSV...", 
                  command=self._handle_file_open).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Y variable:").pack(side=tk.LEFT, padx=(15, 5))
        
        self.y_selector = ttk.Combobox(
            control_frame, textvariable=self.y_choice, 
            state="readonly", width=15
        )
        self.y_selector.pack(side=tk.LEFT)

    def _build_attributes_panel(self, parent):
        """Panel de lista de atributos del dataset"""
        attr_frame = ttk.Labelframe(parent, text="Dataset Attributes", padding=10)
        attr_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        self.attr_tree = ttk.Treeview(attr_frame, columns=("#", "Attribute"), show="headings", height=16)
        self.attr_tree.heading("#", text="#")
        self.attr_tree.heading("Attribute", text="Attribute")
        self.attr_tree.column("#", width=40, anchor="center")
        self.attr_tree.column("Attribute", width=200, anchor="w")
        self.attr_tree.pack(fill=tk.BOTH, expand=True)

    def _build_plots_panel(self, parent):
        """Panel de visualización de gráficas"""
        plots_frame = ttk.Frame(parent)
        plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Frame para gráficas lado a lado
        plots_row = ttk.Frame(plots_frame)
        plots_row.pack(fill=tk.BOTH, expand=True)

        # Gráfica de series temporales
        self.ts_frame = ttk.Frame(plots_row)
        self.ts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Gráfica de correlación
        self.corr_frame = ttk.Frame(plots_row)
        self.corr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Crear gráficas
        self.ts_fig, self.ts_ax, self.ts_canvas = self.plot_manager.create_time_series_plot(self.ts_frame)
        self.corr_fig, self.corr_ax, self.corr_canvas = self.plot_manager.create_correlation_plot(self.corr_frame)

        # Información del dataset
        self._build_dataset_info(plots_frame)

    def _build_dataset_info(self, parent):
        """Panel de información del dataset"""
        info_frame = ttk.Labelframe(parent, text="Dataset Info", padding=10)
        info_frame.pack(fill=tk.X, pady=6)
        self.info_label = ttk.Label(info_frame, text="No dataset loaded")
        self.info_label.pack(anchor="w")

    def _setup_event_handlers(self):
        """Configurar manejadores de eventos"""
        # Actualizar automáticamente cuando cambia la variable Y
        self.y_selector.bind('<<ComboboxSelected>>', self._on_variable_change)

    def _handle_file_open(self):
        """Manejar apertura de archivo CSV"""
        try:
            df = self.data_loader.load_csv()
            if df is not None:
                self._process_loaded_data(df)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n\n{e}")

    def _process_loaded_data(self, df):
        """Procesar datos cargados exitosamente"""
        self.shared_state['current_dataframe'] = df
        self._update_attributes_display(df)
        self._update_variable_selector(df)
        self.info_label.config(text=f"Loaded: {self.data_loader.get_filename()}")
        
        # Plotear matriz de correlación SOLO cuando se carga el CSV
        self._plot_correlation_matrix()

    def _update_attributes_display(self, df):
        """Actualizar lista de atributos"""
        self.attr_tree.delete(*self.attr_tree.get_children())
        for i, col in enumerate(df.columns, 1):
            self.attr_tree.insert("", "end", values=(i, str(col)))

    def _update_variable_selector(self, df):
        """Actualizar selector de variables"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.y_selector['values'] = numeric_cols
        if numeric_cols:
            self.y_choice.set(numeric_cols[0])
            # Actualizar gráfica de series temporales automáticamente
            self._plot_current_data()

    def _plot_correlation_matrix(self):
        """Generar matriz de correlación solo una vez al cargar el CSV"""
        df = self.shared_state.get('current_dataframe')
        if df is None or df.empty:
            return
            
        try:
            # Solo generar matriz de correlación si hay suficientes columnas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                self.plot_manager.plot_correlation_matrix(df[numeric_cols], self.corr_ax, self.corr_canvas)
            else:
                # Limpiar gráfica de correlación si no hay suficientes datos
                self.corr_ax.clear()
                self.corr_ax.set_title("Need at least 2 numeric variables")
                self.corr_canvas.draw()
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")

    def _plot_current_data(self):
        """Generar solo la gráfica de series temporales - llamada automática al cambiar Y"""
        # Verificar que tenemos datos
        df = self.shared_state.get('current_dataframe')
        if df is None or df.empty:
            return
        
        # Verificar que la columna Y seleccionada existe
        y_col = self.y_choice.get()
        if y_col not in df.columns:
            return
        
        # Protección: si ya estamos ploteando, salir
        if hasattr(self, '_is_plotting') and self._is_plotting:
            return
        
        try:
            self._is_plotting = True
            
            # Generar SOLO la gráfica de series temporales
            time_col = self.data_loader.detect_time_column(df)
            self.plot_manager.plot_time_series(df, time_col, y_col, self.ts_ax, self.ts_canvas)
                
        except Exception as e:
            print(f"Error plotting time series: {e}")
        finally:
            self._is_plotting = False

    def _on_variable_change(self, event=None):
        """Manejar cambio de variable seleccionada - actualización automática"""
        # Pequeño delay para asegurar que el cambio se ha procesado
        self.after(100, self._plot_current_data)

    def on_shared_state_change(self, key, value):
        """Manejar cambios en el estado global"""
        if key == 'current_dataframe' and value is not None:
            self._process_loaded_data(value)