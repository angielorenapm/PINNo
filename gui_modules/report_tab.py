#gui_modules/report_tab.py
"""
Módulo para la pestaña de reportes y métricas - USING PLOTLY WITH PROPER DIMENSIONS
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ReportTab(ttk.Frame):
    """Pestaña de reportes y análisis de resultados usando Plotly - PROPER DIMENSIONS"""
    
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.plot_manager = PlotManager()
        
        self._build_interface()
        self._setup_event_handlers()

    def _build_interface(self):
        """Construir la interfaz de la pestaña con Plotly - PROPER DIMENSIONS"""
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
        
        # Gráfica del reporte (derecha) - Now using Plotly with PROPER DIMENSIONS
        self._build_report_plot(main_frame)

    def _build_report_text(self, parent):
        """Construir el área de texto del reporte"""
        text_frame = ttk.Labelframe(parent, text="Analysis Report", padding=8)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        
        # Add scrollbar to text area
        text_container = ttk.Frame(text_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.report_text = tk.Text(text_container, height=20, wrap=tk.WORD)  # Reduced height
        scrollbar = ttk.Scrollbar(text_container, orient="vertical", command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scrollbar.set)
        
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.report_text.insert(tk.END, "Training report will appear here...\n\n")
        self.report_text.insert(tk.END, "• Select a problem and start training\n")
        self.report_text.insert(tk.END, "• Detailed metrics will appear here automatically\n")
        self.report_text.insert(tk.END, "• Comparative plots will be generated\n")
        self.report_text.config(state="disabled")

    def _build_report_plot(self, parent):
        """Construir el área de gráfica del reporte usando Plotly - PROPER DIMENSIONS"""
        plot_frame = ttk.Labelframe(parent, text="Analysis Visualization", padding=8)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))
        
        # Create Plotly figure
        self.report_fig = go.Figure()
        
        # Configure scientific style with PROPER DIMENSIONS
        self.report_fig.update_layout(
            title=dict(
                text="Model Performance Analysis",
                x=0.5,
                xanchor='center',
                font=dict(size=14, family="Arial")  # Smaller font
            ),
            xaxis_title="Domain",
            yaxis_title="Solution",
            template="plotly_white",
            font=dict(family="Arial", size=10),  # Smaller font
            margin=dict(l=50, r=30, t=60, b=40),  # Reduced margins
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            height=400  # Fixed height
        )
        
        # Scientific styling for axes
        self.report_fig.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            showline=True, 
            linewidth=1, 
            linecolor='black',
            mirror=True
        )
        self.report_fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            showline=True, 
            linewidth=1, 
            linecolor='black',
            mirror=True
        )
        
        # Label for displaying Plotly image
        self.plot_label = ttk.Label(plot_frame)
        self.plot_label.pack(fill=tk.BOTH, expand=True)
        
        # Display initial figure
        self._update_plot_display()

    def _update_plot_display(self):
        """Update the Tkinter label with the current Plotly image - PROPER DIMENSIONS"""
        try:
            img_bytes = self.report_fig.to_image(format="png", width=500, height=400, scale=1.5)  # Reduced dimensions
            pil_image = Image.open(io.BytesIO(img_bytes))
            photo = ImageTk.PhotoImage(pil_image)
            self.plot_label.configure(image=photo)
            self.plot_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error updating plot display: {e}")

    def _setup_event_handlers(self):
        """Configurar manejadores de eventos"""
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
            # Load and display the image with PROPER DIMENSIONS
            pil_image = Image.open(path)
            # Resize to fit GUI
            pil_image = pil_image.resize((500, 400), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            self.plot_label.configure(image=photo)
            self.plot_label.image = photo
            
            # Update report text
            self.report_text.config(state="normal")
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, f"External image loaded: {path}\n\n")
            self.report_text.insert(tk.END, "• Image analysis mode\n")
            self.report_text.insert(tk.END, "• Use training to generate PINN-specific reports\n")
            self.report_text.config(state="disabled")
            
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
            self.report_fig.write_image(path, width=600, height=500, scale=2)  # High quality for export
            messagebox.showinfo("Saved", f"Figure saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save figure:\n\n{e}")

    def update_report(self, text_content=None, plot_data=None):
        """Actualizar el contenido del reporte desde otras pestañas"""
        if text_content:
            # Update text content
            self.report_text.config(state="normal")
            self.report_text.delete("1.0", tk.END)
            self.report_text.insert(tk.END, text_content)
            self.report_text.config(state="disabled")
        
        # Update plot if data provided
        if plot_data is not None:
            x, y_true, y_pred = plot_data
            self._update_report_plot(x, y_true, y_pred)

    def _update_report_plot(self, x, y_true, y_pred):
        """Actualizar la gráfica del reporte con Plotly - PROPER DIMENSIONS"""
        try:
            # Clear previous data
            self.report_fig.data = []
            
            # Add analytical solution trace
            self.report_fig.add_trace(go.Scatter(
                x=x,
                y=y_true,
                mode='lines',
                name='Analytical Solution',
                line=dict(color='#d62728', width=2.5, dash='dash'),  # Slightly thinner
                opacity=0.8
            ))
            
            # Add PINN prediction trace
            self.report_fig.add_trace(go.Scatter(
                x=x,
                y=y_pred,
                mode='lines',
                name='PINN Prediction',
                line=dict(color='#2ca02c', width=2),  # Slightly thinner
                opacity=0.9
            ))
            
            # Calculate and display error metrics
            mse = np.mean((y_pred - y_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_pred - y_true))
            
            # Update layout with metrics in subtitle
            self.report_fig.update_layout(
                title=dict(
                    text="PINN vs Analytical Solution Comparison",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=14, family="Arial")  # Smaller font
                ),
                xaxis_title="Domain",
                yaxis_title="Solution Value",
                annotations=[
                    dict(
                        x=0.02,
                        y=0.98,
                        xref="paper",
                        yref="paper",
                        text=f"RMSE: {rmse:.4e} | MAE: {mae:.4e}",
                        showarrow=False,
                        font=dict(size=10, family="Arial"),  # Smaller font
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=4
                    )
                ]
            )
            
            # Update the display
            self._update_plot_display()
            
        except Exception as e:
            print(f"Error updating report plot: {e}")

    def update_comparison_plot(self, x_data_list, y_data_list, labels, title="Model Comparison"):
        """Update plot with multiple datasets for comparison"""
        try:
            # Clear previous data
            self.report_fig.data = []
            
            # Color palette for multiple lines
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Add each dataset
            for i, (x, y, label) in enumerate(zip(x_data_list, y_data_list, labels)):
                color = colors[i % len(colors)]
                self.report_fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=2),  # Slightly thinner
                    opacity=0.8
                ))
            
            # Update layout
            self.report_fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=14, family="Arial")  # Smaller font
                ),
                xaxis_title="Domain",
                yaxis_title="Solution Value"
            )
            
            # Update the display
            self._update_plot_display()
            
        except Exception as e:
            print(f"Error updating comparison plot: {e}")

    def on_shared_state_change(self, key, value):
        """Manejar cambios en el estado global compartido"""
        if key == 'last_metrics_report':
            self.update_report(text_content=value)
        elif key == 'last_plot_data' and value is not None:
            x, y_true, y_pred = value
            self._update_report_plot(x, y_true, y_pred)


class PlotManager:
    """Gestor de gráficas para reportes usando Plotly - PROPER DIMENSIONS"""
    
    def __init__(self):
        self.figures = []

    def create_comparison_plot(self, title="Model Comparison"):
        """Crear gráfica de comparación con estilo científico - PROPER DIMENSIONS"""
        fig = go.Figure()
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=14, family="Arial")  # Smaller font
            ),
            xaxis_title="Domain",
            yaxis_title="Solution Value",
            template="plotly_white",
            font=dict(family="Arial", size=10),  # Smaller font
            margin=dict(l=50, r=30, t=60, b=40),  # Reduced margins
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            ),
            height=400  # Fixed height
        )
        
        # Scientific styling for axes
        fig.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            showline=True, 
            linewidth=1, 
            linecolor='black',
            mirror=True
        )
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='LightGray',
            showline=True, 
            linewidth=1, 
            linecolor='black',
            mirror=True
        )
        
        return fig

    def create_error_analysis_plot(self):
        """Crear gráfica de análisis de errores - PROPER DIMENSIONS"""
        fig = go.Figure()
        
        fig.update_layout(
            title=dict(
                text="Error Analysis",
                x=0.5,
                xanchor='center',
                font=dict(size=14, family="Arial")  # Smaller font
            ),
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            template="plotly_white",
            font=dict(family="Arial", size=10),  # Smaller font
            margin=dict(l=50, r=30, t=60, b=40),  # Reduced margins
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400  # Fixed height
        )
        
        return fig

    def plot_residuals(self, y_true, y_pred, fig):
        """Graficar residuales para análisis de errores"""
        residuals = y_pred - y_true
        
        fig.data = []  # Clear previous data
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                size=5,  # Slightly smaller
                color='#1f77b4',
                opacity=0.6,
                line=dict(width=1, color='DarkSlateGrey')
            )
        ))
        
        # Add zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.8)
        
        fig.update_layout(
            title=dict(text="Residual Analysis", x=0.5),
            xaxis_title="Predicted Values",
            yaxis_title="Residuals (Predicted - True)"
        )
        
        return fig