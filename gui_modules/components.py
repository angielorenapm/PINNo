"""
Componentes reutilizables para la GUI - DataLoader, PlotManager, etc.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
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
    """Gestor de gráficas para la GUI"""
    
    def __init__(self):
        self.figures = []
        self.canvases = []

    def create_time_series_plot(self, parent):
        """Crear gráfica de series temporales"""
        fig = Figure(figsize=(6, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("Time Series")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", linewidth=0.5)
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures.append(fig)
        self.canvases.append(canvas)
        
        return fig, ax, canvas

    def create_correlation_plot(self, parent):
        """Crear gráfica de matriz de correlación"""
        fig = Figure(figsize=(6, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title("Correlation Matrix")
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.figures.append(fig)
        self.canvases.append(canvas)
        
        return fig, ax, canvas

    def plot_time_series(self, df, time_col, value_col, ax, canvas):
        """Graficar series temporales"""
        ax.clear()
        ax.plot(df[time_col], df[value_col], label=value_col)
        ax.set_title(f"{value_col} vs {time_col}")
        ax.set_xlabel(time_col)
        ax.set_ylabel(value_col)
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend()
        fig = ax.get_figure()
        fig.tight_layout()
        canvas.draw()

    def plot_correlation_matrix(self, df, ax, canvas):
        """Graficar matriz de correlación"""
        ax.clear()
        
        if len(df.columns) < 2:
            ax.set_title("Need at least 2 variables for correlation")
            canvas.draw()
            return
        
        corr = df.corr(numeric_only=True)
        im = ax.imshow(corr.values, interpolation="nearest", cmap='coolwarm', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Matrix")
        
        # Añadir valores numéricos
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=9)
        
        fig = ax.get_figure()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        canvas.draw()


class TrainingVisualizer:
    """Visualizador del entrenamiento en tiempo real"""
    
    def __init__(self):
        self.fig = None
        self.ax_loss = None
        self.ax_solution = None
        self.canvas = None
        self.loss_line = None
        self.pred_line = None
        self.true_line = None
        self._current_colorbar = None
        self._original_solution_pos = None  # Store original position

    def setup_plots(self, parent):
        """Configurar las gráficas de entrenamiento"""
        self.fig = Figure(figsize=(10, 6), dpi=100)
        
        # Crear subplots: pérdida arriba, solución abajo
        self.ax_loss = self.fig.add_subplot(2, 1, 1)
        self.ax_solution = self.fig.add_subplot(2, 1, 2)
        
        # Store original position for consistent layout
        self._original_solution_pos = self.ax_solution.get_position()
        
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.init_plots()

    def init_plots(self):
        """Inicializar las gráficas - ONLY called when starting new training"""
        # Gráfica de pérdida
        self.ax_loss.clear()
        self.ax_loss.set_title("Loss Function vs Epochs")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss (log scale)")
        self.ax_loss.grid(True, which="both", linestyle='--', linewidth=0.5)
        self.loss_line, = self.ax_loss.plot([], [], 'b-', linewidth=1.5)
        self.ax_loss.set_yscale('log')

        # Gráfica de solución (inicialmente vacía)
        self.ax_solution.clear()
        self.ax_solution.set_title("Predicted vs Analytical Solution")
        self.ax_solution.grid(True, linestyle='--', linewidth=0.5)
        
        # Inicializar líneas (serán actualizadas según el tipo de problema)
        self.pred_line, = self.ax_solution.plot([], [], 'b-', label="PINN Prediction", linewidth=2)
        self.true_line, = self.ax_solution.plot([], [], 'r--', label="Analytical Solution", linewidth=2)
        self.ax_solution.legend()

        self.canvas.draw()

    def update_loss_plot(self, epoch, loss_history):
        """Actualizar gráfica de pérdida"""
        epochs_data = range(len(loss_history))
        self.loss_line.set_data(epochs_data, loss_history)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        self.canvas.draw_idle()

    def update_solution_plot(self, x, y_true, y_pred, xlabel, ylabel):
        """Actualizar gráfica de solución para problemas 1D"""
        # Limpiar y configurar para 1D
        self.ax_solution.clear()
        self.ax_solution.grid(True, linestyle='--', linewidth=0.5)
        
        # Plotear líneas
        self.ax_solution.plot(x, y_true, 'r--', label="Analytical Solution", linewidth=2)
        self.ax_solution.plot(x, y_pred, 'b-', label="PINN Prediction", linewidth=2)
        
        self.ax_solution.set_xlabel(xlabel)
        self.ax_solution.set_ylabel(ylabel)
        self.ax_solution.set_title("Predicted vs Analytical Solution")
        self.ax_solution.legend()
        
        self.ax_solution.relim()
        self.ax_solution.autoscale_view()
        self.canvas.draw_idle()

    def update_heat_solution_plot(self, X, T, u_pred, title):
        """Actualizar gráfica de solución para calor 2D - FIXED LAYOUT"""
        try:
            # Clear previous plots safely
            self._safe_clear_solution_plot()
            
            # Reset axes position to maintain consistent layout
            self.ax_solution.set_position(self._original_solution_pos)
            
            # Create new contour plot with proper boundaries
            contour = self.ax_solution.contourf(X, T, u_pred, levels=20, cmap='hot')  # Reduced levels
            
            # Create colorbar with proper positioning to avoid layout shifts
            if self._current_colorbar is None:
                # First time: create colorbar with proper spacing
                self._current_colorbar = self.fig.colorbar(
                    contour, ax=self.ax_solution, 
                    label='Temperature (u)',
                    pad=0.05,  # Reduced padding
                    shrink=0.8  # Shrink colorbar to leave space
                )
            else:
                # Update existing colorbar
                self._current_colorbar.mappable.set_array(u_pred)
                self._current_colorbar.update_normal(contour)
            
            # Configure plot with proper boundaries
            self.ax_solution.set_title(title)
            self.ax_solution.set_xlabel('x')
            self.ax_solution.set_ylabel('t')
            self.ax_solution.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Set explicit boundaries to match domain
            self.ax_solution.set_xlim(X.min(), X.max())
            self.ax_solution.set_ylim(T.min(), T.max())
            
            # Adjust layout to prevent shrinking
            self.fig.tight_layout(rect=[0, 0, 0.95, 1])  # Leave 5% space on right for colorbar
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating heat solution plot: {e}")

    def _safe_clear_solution_plot(self):
        """Safely clear the solution plot without colorbar errors"""
        # Clear the main axis
        self.ax_solution.clear()
        
        # Safely remove colorbar if it exists
        if hasattr(self, '_current_colorbar') and self._current_colorbar is not None:
            try:
                # Try to remove the colorbar
                self._current_colorbar.remove()
            except (AttributeError, ValueError) as e:
                # If removal fails, just detach the reference
                pass
            finally:
                # Always clear the reference
                self._current_colorbar = None

    def reset_plots(self):
        """Reset plots completely - only called when starting new training"""
        self._safe_clear_solution_plot()
        
        # Reset to default state
        self.ax_solution.set_title("Predicted vs Analytical Solution")
        self.ax_solution.grid(True, linestyle='--', linewidth=0.5)
        self.ax_solution.legend()
        self.canvas.draw_idle()

    def clear_plots(self):
        """Clear all plots completely - call this when stopping training"""
        self._safe_clear_solution_plot()
        
        # Reset to default state
        self.ax_solution.set_title("Predicted vs Analytical Solution")
        self.ax_solution.grid(True, linestyle='--', linewidth=0.5)
        self.ax_solution.legend()
        self.canvas.draw_idle()


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