#gui_modules/data_explorer.py
"""
Módulo para la pestaña de exploración de datos - Versión corregida con Plotly y DIMENSIONES ADECUADAS
"""
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import io

from gui_modules.components import DataLoader, PlotManager


class DataExplorer(ttk.Frame):
    """Pestaña de exploración y visualización de datos CSV usando Plotly con DIMENSIONES ADECUADAS"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.data_loader = DataLoader()
        self.plot_manager = PlotManager()
        
        # Variables de UI
        self.y_choice = tk.StringVar(value="Select variable")
        self._is_plotting = False
        
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
        """Panel de visualización de gráficas con Plotly - DIMENSIONES ADECUADAS"""
        plots_frame = ttk.Frame(parent)
        plots_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Frame para gráficas lado a lado
        plots_row = ttk.Frame(plots_frame)
        plots_row.pack(fill=tk.BOTH, expand=True)

        # Gráfica de series temporales - TAMAÑO ADECUADO
        self.ts_frame = ttk.Labelframe(plots_row, text="Time Series", padding=5)
        self.ts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.ts_label = ttk.Label(self.ts_frame)
        self.ts_label.pack(fill=tk.BOTH, expand=True)

        # Gráfica de correlación - TAMAÑO ADECUADO
        self.corr_frame = ttk.Labelframe(plots_row, text="Correlation Matrix", padding=5)
        self.corr_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.corr_label = ttk.Label(self.corr_frame)
        self.corr_label.pack(fill=tk.BOTH, expand=True)

        # Información del dataset
        self._build_dataset_info(plots_frame)

    def _build_dataset_info(self, parent):
        """Panel de información del dataset"""
        info_frame = ttk.Labelframe(parent, text="Dataset Info", padding=10)
        info_frame.pack(fill=tk.X, pady=6)
        
        info_text = f"Loaded: {self.data_loader.get_filename()}"
        if hasattr(self.data_loader, 'has_headers'):
            header_status = "With headers" if self.data_loader.has_headers else "No headers (auto-named)"
            info_text += f" | {header_status}"
        
        self.info_label = ttk.Label(info_frame, text=info_text)
        self.info_label.pack(anchor="w")

    def _setup_event_handlers(self):
        """Configurar manejadores de eventos"""
        self.y_selector.bind('<<ComboboxSelected>>', self._on_variable_change)

    def _handle_file_open(self):
        """Manejar apertura de archivo CSV con diálogos de usuario"""
        try:
            df = self.data_loader.load_csv()
            if df is not None:
                # Prompt user for column selection/naming
                df = self.data_loader.prompt_column_selection(self, df)
                if df is not None:
                    self._process_loaded_data(df)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n\n{e}")

    def _process_loaded_data(self, df):
        """Procesar datos cargados exitosamente"""
        self.shared_state['current_dataframe'] = df
        self._update_attributes_display(df)
        self._update_variable_selectors(df)
        
        # Update info with header status and time column
        header_status = "with headers" if self.data_loader.has_header_row() else "without headers (user named)"
        time_col = self.data_loader.get_time_column()
        info_text = f"Loaded: {self.data_loader.get_filename()} | {header_status} | Time: {time_col}"
        self.info_label.config(text=info_text)
        
        # Plotear matriz de correlación SOLO cuando se carga el CSV
        self._plot_correlation_matrix()

    def _update_attributes_display(self, df):
        """Actualizar lista de atributos"""
        self.attr_tree.delete(*self.attr_tree.get_children())
        for i, col in enumerate(df.columns, 1):
            self.attr_tree.insert("", "end", values=(i, str(col)))

    def _update_variable_selectors(self, df):
        """Actualizar selectores de variables (time column is now fixed)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Update Y variable selector (exclude time column)
        time_col = self.data_loader.get_time_column()
        y_options = [col for col in numeric_cols if col != time_col]
        self.y_selector['values'] = y_options
        
        if y_options:
            self.y_choice.set(y_options[0])
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
                # FIXED: Now using correct number of arguments
                fig = self.plot_manager.plot_correlation_matrix(df[numeric_cols])
                self._display_plotly_figure(fig, self.corr_label)
            else:
                # Limpiar gráfica de correlación si no hay suficientes datos
                self.corr_label.config(image='')
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")

    def _plot_current_data(self):
        """Generar solo la gráfica de series temporales - FIXED ARGUMENT COUNT"""
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
            
            # Usar la columna de tiempo seleccionada por el usuario
            time_col = self.data_loader.get_time_column()
            if time_col and time_col in df.columns:
                # FIXED: Now using correct number of arguments
                fig = self.plot_manager.plot_time_series(df, time_col, y_col)
                self._display_plotly_figure(fig, self.ts_label)
            else:
                # Fallback: usar detección automática si no hay columna de tiempo seleccionada
                time_col = self.data_loader.detect_time_column(df)
                fig = self.plot_manager.plot_time_series(df, time_col, y_col)
                self._display_plotly_figure(fig, self.ts_label)
                    
        except Exception as e:
            print(f"Error plotting time series: {e}")
        finally:
            self._is_plotting = False

    def _display_plotly_figure(self, fig, label):
        """Display a Plotly figure in a Tkinter label - DIMENSIONES ADECUADAS"""
        try:
            # Convert Plotly figure to image with PROPER DIMENSIONS
            img_bytes = fig.to_image(format="png", width=400, height=300, scale=1.5)  # Reduced dimensions
            pil_image = Image.open(io.BytesIO(img_bytes))
            photo = ImageTk.PhotoImage(pil_image)
            label.configure(image=photo)
            label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error displaying plot: {e}")

    def _on_variable_change(self, event=None):
        """Manejar cambio de variable seleccionada - actualización automática"""
        self.after(100, self._plot_current_data)

    def on_shared_state_change(self, key, value):
        """Manejar cambios en el estado global"""
        if key == 'current_dataframe' and value is not None:
            self._process_loaded_data(value)