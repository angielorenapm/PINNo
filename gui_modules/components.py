#gui_modules/components.py
"""
Componentes reutilizables para la GUI - DataLoader, PlotManager, etc.
USING PLOTLY WITH PROPER DIMENSIONS FOR GUI
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox  
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image, ImageTk
import io
import tensorflow as tf


class DataLoader:
    """Cargador y validador de datos CSV con detección automática de headers"""
    
    def __init__(self):
        self.current_filename = None
        self.has_headers = True  # Default assumption
        self.column_names = []   # Store detected column names
        self.time_column = None  # User-selected time column

    def load_csv(self):
        """Cargar archivo CSV y detectar automáticamente si tiene headers"""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return None
        
        try:
            # First attempt: try reading with headers
            try:
                df_with_headers = pd.read_csv(path)
                self.has_headers = True
                self.column_names = df_with_headers.columns.tolist()
                df = df_with_headers
            except Exception:
                # If reading with headers fails, try without headers
                df = pd.read_csv(path, header=None)
                self.has_headers = False
                self.column_names = [f'Column_{i+1}' for i in range(len(df.columns))]
                df.columns = self.column_names
            
            self.current_filename = path.split("/")[-1]
            
            # Validación básica: al menos una columna numérica
            if df.select_dtypes(include=[np.number]).empty:
                messagebox.showerror("Invalid File", "CSV must contain at least one numeric column.")
                return None
                
            return df
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n\n{e}")
            return None

    def prompt_column_selection(self, parent, df):
        """Prompt user to select time column and optionally name columns"""
        if not self.has_headers:
            # No headers - ask user to name columns
            column_names = self._prompt_column_names(parent, df)
            if column_names is None:  # User cancelled
                return None
            df.columns = column_names
            self.column_names = column_names
        
        # Ask user to select time column
        time_col = self._prompt_time_column(parent, df)
        if time_col is None:  # User cancelled
            return None
        
        self.time_column = time_col
        return df

    def _prompt_column_names(self, parent, df):
        """Dialog for user to name columns when no headers exist"""
        # Create dialog window
        dialog = tk.Toplevel(parent)
        dialog.title("Name Your Columns")
        dialog.geometry("400x300")
        dialog.transient(parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Please name your data columns:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Frame for column entries
        entry_frame = ttk.Frame(dialog)
        entry_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        entries = []
        default_names = [f'Column_{i+1}' for i in range(len(df.columns))]
        
        for i, default_name in enumerate(default_names):
            ttk.Label(entry_frame, text=f"Column {i+1}:").grid(row=i, column=0, sticky='w', pady=5)
            entry = ttk.Entry(entry_frame, width=20)
            entry.insert(0, default_name)
            entry.grid(row=i, column=1, sticky='ew', padx=5, pady=5)
            entries.append(entry)
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        result = [None]  # Store result in list to modify in closure
        
        def on_ok():
            names = [entry.get().strip() for entry in entries]
            # Validate: no empty names and no duplicates
            if any(not name for name in names):
                messagebox.showerror("Error", "Column names cannot be empty.", parent=dialog)
                return
            if len(names) != len(set(names)):
                messagebox.showerror("Error", "Column names must be unique.", parent=dialog)
                return
            result[0] = names
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        
        dialog.wait_window()
        return result[0]

    def _prompt_time_column(self, parent, df):
        """Dialog for user to select time column"""
        # Create dialog window
        dialog = tk.Toplevel(parent)
        dialog.title("Select Time Column")
        dialog.geometry("400x200")
        dialog.transient(parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Please select the time/independent variable column:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        # Time column selection
        time_var = tk.StringVar()
        time_frame = ttk.Frame(dialog)
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(time_frame, text="Time column:").pack(side=tk.LEFT)
        time_combo = ttk.Combobox(time_frame, textvariable=time_var, 
                                 values=self.column_names, state="readonly")
        time_combo.pack(side=tk.LEFT, padx=10)
        time_combo.set(self.column_names[0])  # Default to first column
        
        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        result = [None]  # Store result in list to modify in closure
        
        def on_ok():
            result[0] = time_var.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        
        dialog.wait_window()
        return result[0]

    def detect_time_column(self, df):
        """Intentar detectar la columna de tiempo automáticamente (fallback)"""
        time_indicators = ['t', 'time', 't(s)', 't_s', 'timestamp', 'column_1', 'col_1']
        
        # First, check actual column names
        for col in df.columns:
            if str(col).lower() in time_indicators:
                return col
        
        # If no match, use heuristics based on data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Prefer the first column that looks like time data
        for col in numeric_cols:
            col_data = df[col]
            # Check if data looks like time (monotonically increasing, starts near 0)
            if len(col_data) > 1:
                diffs = np.diff(col_data)
                if np.all(diffs >= 0) and col_data.iloc[0] >= 0:
                    return col
        
        # Fallback: use the first numeric column
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

    def get_filename(self):
        """Obtener el nombre del archivo actual"""
        return self.current_filename

    def get_column_names(self):
        """Obtener los nombres de columnas detectados"""
        return self.column_names

    def get_time_column(self):
        """Obtener la columna de tiempo seleccionada"""
        return self.time_column

    def has_header_row(self):
        """Indicar si el archivo tiene headers"""
        return self.has_headers


class PlotManager:
    """Gestor de gráficas para la GUI usando Plotly - PROPER DIMENSIONS"""
    
    def __init__(self):
        self.figures = []

    def create_time_series_plot(self):
        """Crear gráfica de series temporales con Plotly - SMALLER DIMENSIONS"""
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Time Series", x=0.5, xanchor='center', font=dict(size=14, family="Arial")),
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            font=dict(family="Arial", size=10),
            margin=dict(l=50, r=30, t=50, b=40),  # Reduced margins
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300  # Fixed height
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='black')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', showline=True, linewidth=1, linecolor='black')
        return fig

    def create_correlation_plot(self):
        """Crear gráfica de matriz de correlación con Plotly - SMALLER DIMENSIONS"""
        fig = go.Figure()
        fig.update_layout(
            title=dict(text="Correlation Matrix", x=0.5, xanchor='center', font=dict(size=14, family="Arial")),
            template="plotly_white",
            font=dict(family="Arial", size=10),
            margin=dict(l=50, r=30, t=50, b=40),  # Reduced margins
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300  # Fixed height
        )
        return fig

    def plot_time_series(self, df, time_col, value_col):
        """Graficar series temporales con Plotly - FIXED: Now takes 3 arguments"""
        fig = self.create_time_series_plot()
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[value_col],
            mode='lines',
            name=value_col,
            line=dict(width=2, color='#1f77b4')
        ))
        fig.update_layout(
            title=dict(text=f"{value_col} vs {time_col}", x=0.5, xanchor='center', font=dict(size=14)),
            xaxis_title=time_col,
            yaxis_title=value_col
        )
        return fig

    def plot_correlation_matrix(self, df):
        """Graficar matriz de correlación con Plotly - FIXED: Now takes 1 argument"""
        if len(df.columns) < 2:
            fig = self.create_correlation_plot()
            fig.update_layout(title=dict(text="Need at least 2 variables for correlation", x=0.5))
            return fig
        
        corr = df.corr(numeric_only=True)
        
        fig = self.create_correlation_plot()
        fig.add_trace(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            text=[[f'{val:.2f}' for val in row] for row in corr.values],
            texttemplate="%{text}",
            textfont={"size": 8, "color": "black"}  # Smaller font
        ))
        
        fig.update_layout(
            title=dict(text="Correlation Matrix", x=0.5, xanchor='center'),
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        return fig


class TrainingVisualizer:
    """Visualizador del entrenamiento en tiempo real usando Plotly - PROPER DIMENSIONS"""
    
    def __init__(self):
        self.fig = None
        self.loss_history = []
        self.current_problem = None

    def setup_plots(self):
        """Configurar las gráficas de entrenamiento con Plotly - SMALLER DIMENSIONS"""
        # Create subplot figure with proper sizing
        self.fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss Function vs Epochs', 'Predicted vs Analytical Solution'),
            vertical_spacing=0.12,  # Adjusted spacing
            row_heights=[0.4, 0.6]
        )
        
        # Configure layout with proper dimensions for GUI
        self.fig.update_layout(
            height=450,  # Reduced from 700 to fit GUI
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial", size=10),  # Smaller font
            margin=dict(l=50, r=30, t=60, b=40),  # Reduced margins
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Scientific styling for axes
        self.fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', 
                             showline=True, linewidth=1, linecolor='black')
        self.fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', 
                             showline=True, linewidth=1, linecolor='black')
        
        # Loss plot configuration
        self.fig.update_xaxes(title_text="Epoch", row=1, col=1)
        self.fig.update_yaxes(title_text="Loss (log scale)", type="log", row=1, col=1)
        
        # Solution plot configuration
        self.fig.update_xaxes(title_text="Domain", row=2, col=1)
        self.fig.update_yaxes(title_text="Solution", row=2, col=1)

    def init_plots(self):
        """Inicializar las gráficas - ONLY called when starting new training"""
        # Clear all data
        self.fig.data = []
        self.loss_history = []
        
        # Add empty traces for loss plot
        self.fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name='Training Loss',
            line=dict(color='#1f77b4', width=2)
        ), row=1, col=1)

    def update_loss_plot(self, epoch, loss_history):
        """Actualizar gráfica de pérdida"""
        self.loss_history = loss_history
        epochs = list(range(len(loss_history)))
        
        # Update loss trace
        self.fig.data[0].x = epochs
        self.fig.data[0].y = loss_history

    def update_solution_plot(self, x, y_true, y_pred, xlabel, ylabel):
        """Actualizar gráfica de solución para problemas 1D - PROPER DIMENSIONS"""
        # Clear previous solution data (keep loss data)
        current_data = list(self.fig.data)
        if len(current_data) > 1:
            self.fig.data = current_data[:1]  # Keep only loss trace
        
        # Add new solution traces with proper sizing
        self.fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=y_true.flatten(),
            mode='lines',
            name='Analytical Solution',
            line=dict(color='#d62728', width=2, dash='dash'),  # Slightly thinner lines
            showlegend=True
        ), row=2, col=1)
        
        self.fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=y_pred.flatten(),
            mode='lines',
            name='PINN Prediction',
            line=dict(color='#2ca02c', width=2),  # Slightly thinner lines
            showlegend=True
        ), row=2, col=1)
        
        # Update axes labels
        self.fig.update_xaxes(title_text=xlabel, row=2, col=1)
        self.fig.update_yaxes(title_text=ylabel, row=2, col=1)
        
        # Update subplot title
        self.fig.update_layout(
            title_text="Training Progress",
            title_x=0.5,
            title_font=dict(size=14, family="Arial")  # Smaller title
        )

    def update_heat_solution_plot(self, X, T, u_pred, title):
        """Actualizar gráfica de solución para calor 2D - PROPER DIMENSIONS - FIXED COLORBAR"""
        try:
            # Clear previous solution data (keep loss data)
            current_data = list(self.fig.data)
            if len(current_data) > 1:
                self.fig.data = current_data[:1]  # Keep only loss trace
            
            # Create heatmap with proper sizing - FIXED COLORBAR CONFIGURATION
            self.fig.add_trace(go.Heatmap(
                x=X[0, :],  # x coordinates (first row)
                y=T[:, 0],  # t coordinates (first column)
                z=u_pred,
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(
                        text='Temperature (u)',
                        side='right'  # FIXED: Changed from 'titleside' to 'side'
                    ),
                    title_font=dict(size=10)  # FIXED: Updated property name
                ),
                name='PINN Solution',
                showscale=True,
                showlegend=False
            ), row=2, col=1)
            
            # Update layout with proper sizing
            self.fig.update_xaxes(title_text='x', row=2, col=1)
            self.fig.update_yaxes(title_text='t', row=2, col=1)
            self.fig.update_layout(
                title_text=title,
                title_x=0.5,
                title_font=dict(size=12, family="Arial")  # Smaller title
            )
            
        except Exception as e:
            print(f"Error updating heat solution plot: {e}")

    def reset_plots(self):
        """Reset plots completely"""
        self.fig.data = []
        self.loss_history = []
        self.init_plots()

    def clear_plots(self):
        """Clear all plots completely"""
        self.fig.data = []
        self.loss_history = []

    def get_plot_image(self):
        """Convert Plotly figure to PIL Image for display in Tkinter - PROPER DIMENSIONS"""
        try:
            # Convert Plotly figure to image with PROPER DIMENSIONS
            img_bytes = self.fig.to_image(format="png", width=600, height=450, scale=1.5)  # Reduced dimensions
            
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(img_bytes))
            return pil_image
            
        except Exception as e:
            print(f"Error converting plot to image: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (600, 450), color='white')


class MetricsCalculator:
    """Calculador de métricas para evaluar el modelo"""
    
    def __init__(self):
        pass

    def comprehensive_report(self, y_true, y_pred):
        """Generar reporte completo de métricas"""
        n = len(y_true)
        if n == 0:
            return "No data for metrics calculation."

        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))

        # Errores relativos
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

        return "\n".join(lines)

    def basic_metrics(self, y_true, y_pred):
        """Métricas básicas de regresión"""
        n = len(y_true)
        if n == 0:
            return {}
            
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err**2))
        rmse = float(np.sqrt(mse))
        
        # Error relativo
        y_norm = np.linalg.norm(y_true)
        relative_error = np.linalg.norm(err) / (y_norm + 1e-12)
        
        return {
            'MAE': mae,
            'MSE': mse, 
            'RMSE': rmse,
            'Relative Error': relative_error,
            'n_samples': n
        }