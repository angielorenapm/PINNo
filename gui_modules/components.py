"""
Componentes reutilizables para la GUI - DataLoader, PlotManager, etc.
USING PLOTLY WITH PROPER DIMENSIONS FOR GUI
(Versión 0.0.4 - Con soporte para datos CSV)
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
        self.has_headers = True
        self.column_names = []
        self.time_column = None

    def load_csv(self):
        """Cargar archivo CSV y detectar automáticamente si tiene headers"""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return None
        
        try:
            try:
                df_with_headers = pd.read_csv(path)
                self.has_headers = True
                self.column_names = df_with_headers.columns.tolist()
                df = df_with_headers
            except Exception:
                df = pd.read_csv(path, header=None)
                self.has_headers = False
                self.column_names = [f'Column_{i+1}' for i in range(len(df.columns))]
                df.columns = self.column_names
            
            self.current_filename = path.split("/")[-1]
            
            if df.select_dtypes(include=[np.number]).empty:
                messagebox.showerror("Invalid File", "CSV must contain at least one numeric column.")
                return None
                
            return df
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n\n{e}")
            return None

    def prompt_column_selection(self, parent, df, problem_name=None):
        """Prompt user to select columns with problem-specific guidance"""
        if not self.has_headers:
            column_names = self._prompt_column_names(parent, df, problem_name)
            if column_names is None:
                return None
            df.columns = column_names
            self.column_names = column_names
        
        if problem_name is not None:
            if not self._prompt_problem_columns(parent, df, problem_name):
                return None
        else:
            time_col = self._prompt_time_column(parent, df)
            if time_col is None:
                return None
            self.time_column = time_col
        
        return df

    def _prompt_problem_columns(self, parent, df, problem_name: str) -> bool:
        """Dialog for user to select problem-specific columns"""
        from src.config import get_csv_columns
        
        required_cols = get_csv_columns(problem_name)
        
        dialog = tk.Toplevel(parent)
        dialog.title(f"Select Columns for {problem_name}")
        dialog.geometry("500x300")
        dialog.transient(parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text=f"Please map columns for {problem_name}:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        selection_frame = ttk.Frame(dialog)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        selections = {}
        
        for i, col_name in enumerate(required_cols):
            ttk.Label(selection_frame, text=f"{col_name}:").grid(row=i, column=0, sticky='w', pady=5)
            var = tk.StringVar()
            combo = ttk.Combobox(selection_frame, textvariable=var, 
                                values=self.column_names, state="readonly")
            combo.grid(row=i, column=1, sticky='ew', padx=5, pady=5)
            # Set default selection
            if i < len(self.column_names):
                combo.set(self.column_names[i])
            selections[col_name] = var
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        result = [None]
        
        def on_ok():
            selected = {col: var.get() for col, var in selections.items()}
            if any(not value for value in selected.values()):
                messagebox.showerror("Error", "Please select a column for each field.", parent=dialog)
                return
            if len(set(selected.values())) != len(selected):
                messagebox.showerror("Error", "Column selections must be unique.", parent=dialog)
                return
            # Store the selections
            for col, selected_col in selected.items():
                setattr(self, f"{col}_column", selected_col)
            result[0] = True
            dialog.destroy()
        
        def on_cancel():
            result[0] = False
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=5)
        
        dialog.wait_window()
        return result[0]

    def _prompt_column_names(self, parent, df, problem_name=None):
        """Dialog for user to name columns when no headers exist"""
        dialog = tk.Toplevel(parent)
        dialog.title("Name Your Columns")
        dialog.geometry("400x300")
        dialog.transient(parent)
        dialog.grab_set()
        
        title_text = "Please name your data columns:"
        if problem_name:
            title_text += f"\n(Recommended for {problem_name}: {', '.join(get_csv_columns(problem_name))})"
            
        ttk.Label(dialog, text=title_text, 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
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
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        result = [None]
        
        def on_ok():
            names = [entry.get().strip() for entry in entries]
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
        dialog = tk.Toplevel(parent)
        dialog.title("Select Time Column")
        dialog.geometry("400x200")
        dialog.transient(parent)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Please select the time/independent variable column:", 
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
        time_var = tk.StringVar()
        time_frame = ttk.Frame(dialog)
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(time_frame, text="Time column:").pack(side=tk.LEFT)
        time_combo = ttk.Combobox(time_frame, textvariable=time_var, 
                                 values=self.column_names, state="readonly")
        time_combo.pack(side=tk.LEFT, padx=10)
        time_combo.set(self.column_names[0])
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        result = [None]
        
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
        """Intentar detectar la columna de tiempo automáticamente"""
        time_indicators = ['t', 'time', 't(s)', 't_s', 'timestamp', 'column_1', 'col_1']
        
        for col in df.columns:
            if str(col).lower() in time_indicators:
                return col
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col]
            if len(col_data) > 1:
                diffs = np.diff(col_data)
                if np.all(diffs >= 0) and col_data.iloc[0] >= 0:
                    return col
        
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

    def get_filename(self):
        return self.current_filename

    def get_column_names(self):
        return self.column_names

    def get_time_column(self):
        return self.time_column

    def has_header_row(self):
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
            margin=dict(l=50, r=30, t=50, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
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
            margin=dict(l=50, r=30, t=50, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300
        )
        return fig

    def plot_time_series(self, df, time_col, value_col):
        """Graficar series temporales con Plotly"""
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
        """Graficar matriz de correlación con Plotly"""
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
            textfont={"size": 8, "color": "black"}
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
        self.fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Loss Function vs Epochs', 'Predicted vs Analytical Solution'),
            vertical_spacing=0.12,
            row_heights=[0.4, 0.6]
        )
        
        self.fig.update_layout(
            height=450,
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial", size=10),
            margin=dict(l=50, r=30, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        self.fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', 
                             showline=True, linewidth=1, linecolor='black')
        self.fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', 
                             showline=True, linewidth=1, linecolor='black')
        
        self.fig.update_xaxes(title_text="Epoch", row=1, col=1)
        self.fig.update_yaxes(title_text="Loss (log scale)", type="log", row=1, col=1)
        
        self.fig.update_xaxes(title_text="Domain", row=2, col=1)
        self.fig.update_yaxes(title_text="Solution", row=2, col=1)

    def init_plots(self):
        """Inicializar las gráficas - ONLY called when starting new training"""
        self.fig.data = []
        self.loss_history = []
        
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
        
        self.fig.data[0].x = epochs
        self.fig.data[0].y = loss_history

    def update_solution_plot(self, x, y_true, y_pred, xlabel, ylabel):
        """Actualizar gráfica de solución para problemas 1D - PROPER DIMENSIONS"""
        current_data = list(self.fig.data)
        if len(current_data) > 1:
            self.fig.data = current_data[:1]
        
        self.fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=y_true.flatten(),
            mode='lines',
            name='True Solution',
            line=dict(color='#d62728', width=2, dash='dash'),
            showlegend=True
        ), row=2, col=1)
        
        self.fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=y_pred.flatten(),
            mode='lines',
            name='PINN Prediction',
            line=dict(color='#2ca02c', width=2),
            showlegend=True
        ), row=2, col=1)
        
        self.fig.update_xaxes(title_text=xlabel, row=2, col=1)
        self.fig.update_yaxes(title_text=ylabel, row=2, col=1)
        
        self.fig.update_layout(
            title_text="Training Progress",
            title_x=0.5,
            title_font=dict(size=14, family="Arial")
        )

    def update_heat_solution_plot(self, X, T, u_pred, title):
        """Actualizar gráfica de solución para calor 2D - PROPER DIMENSIONS"""
        try:
            current_data = list(self.fig.data)
            if len(current_data) > 1:
                self.fig.data = current_data[:1]
            
            # FIXED: Use 'side' instead of 'titleside'
            self.fig.add_trace(go.Heatmap(
                x=X[0, :],
                y=T[:, 0],
                z=u_pred,
                colorscale='Viridis',
                colorbar=dict(
                    title='Temperature (u)',
                    side='right',  # CORRECTED: titleside -> side
                    titlefont=dict(size=10)
                ),
                name='PINN Solution',
                showscale=True,
                showlegend=False
            ), row=2, col=1)
            
            self.fig.update_xaxes(title_text='x', row=2, col=1)
            self.fig.update_yaxes(title_text='t', row=2, col=1)
            self.fig.update_layout(
                title_text=title,
                title_x=0.5,
                title_font=dict(size=12, family="Arial")
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
            img_bytes = self.fig.to_image(format="png", width=600, height=450, scale=1.5)
            pil_image = Image.open(io.BytesIO(img_bytes))
            return pil_image
            
        except Exception as e:
            print(f"Error converting plot to image: {e}")
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

        y_bar = float(np.mean(y_true))
        rae = float(np.sum(np.abs(err)) / (np.sum(np.abs(y_true - y_bar)) + 1e-12)) * 100.0
        rrse = float(np.sqrt(np.sum(err**2) / (np.sum((y_true - y_bar)**2) + 1e-12))) * 100.0

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

        prec_a, rec_a, f1_a = prf(tn, fn, fp)
        prec_b, rec_b, f1_b = prf(tp, fp, fn)

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
        
        y_norm = np.linalg.norm(y_true)
        relative_error = np.linalg.norm(err) / (y_norm + 1e-12)
        
        return {
            'MAE': mae,
            'MSE': mse, 
            'RMSE': rmse,
            'Relative Error': relative_error,
            'n_samples': n
        }