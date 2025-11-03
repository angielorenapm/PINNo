"""
Componentes reutilizables para la GUI - DataLoader, PlotManager, etc.
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf


class DataLoader:
    """Cargador y validador de datos CSV"""
    
    def __init__(self):
        self.current_filename = None

    def load_csv(self):
        """Cargar archivo CSV y validar formato básico"""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return None
        
        try:
            df = pd.read_csv(path)
            self.current_filename = path.split("/")[-1]
            
            # Validación básica: al menos una columna numérica
            if df.select_dtypes(include=[np.number]).empty:
                messagebox.showerror("Invalid File", "CSV must contain at least one numeric column.")
                return None
                
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n\n{e}")
            return None

    def detect_time_column(self, df):
        """Intentar detectar la columna de tiempo automáticamente"""
        time_indicators = ['t', 'time', 't(s)', 't_s', 'timestamp']
        for col in df.columns:
            if col.lower() in time_indicators:
                return col
        # Si no se encuentra, usar la primera columna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]

    def get_filename(self):
        """Obtener el nombre del archivo actual"""
        return self.current_filename


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
        """Actualizar gráfica de solución para calor 2D - SINGLE COLORBAR APPROACH"""
        try:
            # Clear only the contour, not the entire axis
            for collection in self.ax_solution.collections:
                collection.remove()
            
            # Create new contour plot
            contour = self.ax_solution.contourf(X, T, u_pred, levels=20, cmap='hot')
            
            # Update or create colorbar
            if self._current_colorbar is None:
                # First time: create colorbar
                self._current_colorbar = self.fig.colorbar(
                    contour, ax=self.ax_solution, 
                    label='Temperature (u)',
                    pad=0.05
                )
            else:
                # Update existing colorbar
                self._current_colorbar.update_normal(contour)
            
            # Configure plot
            self.ax_solution.set_title(title)
            self.ax_solution.set_xlabel('x')
            self.ax_solution.set_ylabel('t')
            self.ax_solution.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
            
            # Set boundaries
            self.ax_solution.set_xlim(X.min(), X.max())
            self.ax_solution.set_ylim(T.min(), T.max())
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating heat solution plot: {e}")
            # Fallback: complete clear and redraw
            self._safe_clear_solution_plot()
            contour = self.ax_solution.contourf(X, T, u_pred, levels=20, cmap='hot')
            self._current_colorbar = self.fig.colorbar(contour, ax=self.ax_solution, label='Temperature (u)')
            self.ax_solution.set_title(title)
            self.ax_solution.set_xlabel('x')
            self.ax_solution.set_ylabel('t')
            self.canvas.draw_idle()

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