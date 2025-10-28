# gui_modules/report_tab.py
"""
Módulo para la pestaña de reportes y métricas
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ReportTab(ttk.Frame):
    """Pestaña de reportes y análisis de resultados"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        
        self._build_interface()
        self._setup_event_handlers()

    def _build_interface(self):
        """Construir la interfaz de la pestaña"""
        # Panel superior de controles
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(control_frame, text="Open Report Image...", 
                  command=self._open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Report (.txt)", 
                  command=self._save_report_txt).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Figure (.png)", 
                  command=self._save_report_png).pack(side=tk.LEFT, padx=5)

        # Panel principal
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Área de texto del reporte (izquierda)
        self._build_report_text(main_frame)
        
        # Gráfica del reporte (derecha)
        self._build_report_plot(main_frame)

    def _build_report_text(self, parent):
        """Construir el área de texto del reporte"""
        text_frame = ttk.Labelframe(parent, text="Report", padding=8)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        
        self.report_text = tk.Text(text_frame, height=25)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self.report_text.insert(tk.END, "Training report will appear here...\n")
        self.report_text.config(state="disabled")

    def _build_report_plot(self, parent):
        """Construir el área de gráfica del reporte"""
        plot_frame = ttk.Labelframe(parent, text="Report Plot", padding=8)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))
        
        self.report_fig = plt.Figure(figsize=(6, 4.6), dpi=100)
        self.report_ax = self.report_fig.add_subplot(111)
        self.report_ax.set_title("No plot yet")
        self.report_ax.grid(True, linestyle="--", linewidth=0.5)
        
        self.report_canvas = FigureCanvasTkAgg(self.report_fig, master=plot_frame)
        self.report_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_event_handlers(self):
        """Configurar manejadores de eventos"""
        # Por ahora, no hay eventos específicos
        pass

    def _open_image(self):
        """Abrir una imagen externa para el reporte"""
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
            messagebox.showerror("Error", f"Failed to load image:\n\n{e}")

    def _save_report_txt(self):
        """Guardar el reporte como texto"""
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
            messagebox.showerror("Error", f"Failed to save report:\n\n{e}")

    def _save_report_png(self):
        """Guardar la gráfica como PNG"""
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
            messagebox.showerror("Error", f"Failed to save figure:\n\n{e}")

    def update_report(self, text_content, plot_data=None):
        """Actualizar el contenido del reporte"""
        # Actualizar texto
        self.report_text.config(state="normal")
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, text_content)
        self.report_text.config(state="disabled")
        
        # Actualizar gráfica si se proporcionan datos
        if plot_data is not None:
            x, y_true, y_pred = plot_data
            self.report_ax.clear()
            self.report_ax.plot(x, y_true, 'r--', label="Analytical")
            self.report_ax.plot(x, y_pred, 'b-', label="Predicted")
            self.report_ax.set_title("Predicted vs Analytical Solution")
            self.report_ax.set_xlabel("Domain")
            self.report_ax.set_ylabel("Magnitude")
            self.report_ax.grid(True, linestyle="--", linewidth=0.5)
            self.report_ax.legend()
            self.report_fig.tight_layout()
            self.report_canvas.draw()

    def on_state_change(self, key, value):
        """Manejar cambios en el estado global"""
        # Por ejemplo, cuando termina el entrenamiento, se podría actualizar el reporte
        if key == 'trainer' and value is not None:
            # Aquí se podría generar un reporte automáticamente
            pass

    def update_report(self, text_content=None, plot_data=None):
        """Actualizar el contenido del reporte desde otras pestañas"""
        if text_content:
            # Actualizar texto
            self.report_text.config(state="normal")
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, text_content)
            self.report_text.config(state="disabled")
        
        # Actualizar gráfica si se proporcionan datos
        if plot_data is not None:
            x, y_true, y_pred = plot_data
            self._update_report_plot(x, y_true, y_pred)

    def _update_report_plot(self, x, y_true, y_pred):
        """Actualizar la gráfica del reporte"""
        try:
            self.report_ax.clear()
            self.report_ax.plot(x, y_true, 'r--', label="Analytical Solution")
            self.report_ax.plot(x, y_pred, 'b-', label="PINN Prediction", linewidth=2)
            self.report_ax.set_title("Predicted vs Analytical Solution")
            self.report_ax.set_xlabel("Domain")
            self.report_ax.set_ylabel("Solution")
            self.report_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            self.report_ax.legend()
            self.report_fig.tight_layout()
            self.report_canvas.draw()
        except Exception as e:
            print(f"Error updating report plot: {e}")

    def on_shared_state_change(self, key, value):
        """Manejar cambios en el estado global compartido"""
        if key == 'last_metrics_report':
            self.update_report(text_content=value)
        elif key == 'last_plot_data' and value is not None:
            x, y_true, y_pred = value
            self._update_report_plot(x, y_true, y_pred)