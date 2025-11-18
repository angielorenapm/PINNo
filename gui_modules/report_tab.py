# src/gui_modules/report_tab.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd  # Necesario para exportar a CSV
import json

class ReportTab(ttk.Frame):
    """
    Pestaña de Reportes y Métricas Finales.
    Permite visualizar resultados y exportar Modelos, Gráficas y Datos.
    """
    def __init__(self, parent, shared_state):
        super().__init__(parent)
        self.shared_state = shared_state
        self.training_tab_ref = None
        
        self._setup_ui()

    def _setup_ui(self):
        # Layout: Panel Lateral (Métricas + Exportar) | Panel Principal (Gráfica)
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # --- PANEL IZQUIERDO ---
        left_panel = ttk.Frame(self.main_container, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Título
        ttk.Label(left_panel, text="Resultados del Entrenamiento", 
                 font=("Helvetica", 14, "bold")).pack(pady=(0, 15), anchor="w")

        # Botón Actualizar
        self.btn_refresh = ttk.Button(left_panel, text="🔄 Generar Reporte", command=self.generate_report)
        self.btn_refresh.pack(fill="x", pady=(0, 20))

        # Sección 1: Métricas
        self.stats_frame = ttk.LabelFrame(left_panel, text="Métricas Clave")
        self.stats_frame.pack(fill="x", pady=5, ipady=5)
        
        self.lbl_final_loss = self._add_stat_row("Pérdida Final:", "---")
        self.lbl_min_loss = self._add_stat_row("Pérdida Mínima:", "---")
        self.lbl_epochs = self._add_stat_row("Épocas Totales:", "---")
        self.lbl_problem = self._add_stat_row("Problema:", "---")

        # Sección 2: Exportación
        self.export_frame = ttk.LabelFrame(left_panel, text="Exportar Resultados")
        self.export_frame.pack(fill="x", pady=20, ipady=5)
        
        self.btn_save_plot = ttk.Button(self.export_frame, text="💾 Guardar Gráfica (.png)", command=self.save_plot)
        self.btn_save_plot.pack(fill="x", padx=5, pady=2)
        
        self.btn_save_csv = ttk.Button(self.export_frame, text="📄 Guardar Datos (.csv)", command=self.save_data)
        self.btn_save_csv.pack(fill="x", padx=5, pady=2)
        
        self.btn_save_model = ttk.Button(self.export_frame, text="🧠 Guardar Modelo (.keras)", command=self.save_model)
        self.btn_save_model.pack(fill="x", padx=5, pady=2)
        
        # Estado inicial botones deshabilitados
        self._toggle_export_buttons(False)

        # Sección 3: Configuración (Solo lectura)
        self.config_frame = ttk.LabelFrame(left_panel, text="Configuración Usada")
        # --- CORRECCIÓN AQUÍ: 'fill' solo se usa una vez ---
        self.config_frame.pack(fill="both", expand=True, pady=10) 
        
        self.txt_config = tk.Text(self.config_frame, height=8, width=30, font=("Consolas", 9), bg="#f4f6f6")
        self.txt_config.pack(fill="both", expand=True, padx=5, pady=5)
        self.txt_config.insert("1.0", "Entrena un modelo para ver detalles.")
        self.txt_config.config(state="disabled")

        # --- PANEL DERECHO: Gráfica ---
        right_panel = ttk.Frame(self.main_container)
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor('#f0f2f5')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        self._reset_plot()

    def _add_stat_row(self, label, value):
        row = ttk.Frame(self.stats_frame)
        row.pack(fill="x", padx=10, pady=3)
        ttk.Label(row, text=label, font=("Helvetica", 9, "bold"), width=15).pack(side="left")
        lbl_val = ttk.Label(row, text=value, font=("Helvetica", 9))
        lbl_val.pack(side="right")
        return lbl_val

    def _toggle_export_buttons(self, state: bool):
        s = "normal" if state else "disabled"
        self.btn_save_plot.config(state=s)
        self.btn_save_csv.config(state=s)
        self.btn_save_model.config(state=s)

    def _reset_plot(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "No hay datos disponibles.\nEntrena un modelo y pulsa 'Generar Reporte'.", 
                    ha='center', va='center', color='#7f8c8d')
        self.ax.axis('off')
        self.canvas.draw()

    def generate_report(self):
        trainer = self.shared_state.get('trainer')
        
        if not trainer or not trainer.loss_history:
            self._reset_plot()
            self._toggle_export_buttons(False)
            return

        # Habilitar botones
        self._toggle_export_buttons(True)

        # 1. Datos
        history = np.array(trainer.loss_history)
        epochs = len(history)
        final_loss = history[-1]
        min_loss = np.min(history)
        problem = self.shared_state.get('problem_name', tk.StringVar(value="?")).get()
        
        # 2. Etiquetas
        self.lbl_final_loss.config(text=f"{final_loss:.2e}")
        self.lbl_min_loss.config(text=f"{min_loss:.2e}")
        self.lbl_epochs.config(text=str(epochs))
        self.lbl_problem.config(text=problem)

        # 3. Configuración
        self.txt_config.config(state="normal")
        self.txt_config.delete("1.0", tk.END)
        config_str = self._format_config_text(trainer.config, problem)
        self.txt_config.insert("1.0", config_str)
        self.txt_config.config(state="disabled")

        # 4. Gráfica
        self.ax.clear()
        self.ax.axis('on')
        self.ax.plot(history, label='Total Loss', color='#2980b9', linewidth=1.5)
        
        if len(history) > 50:
            window = max(int(len(history)/50), 5)
            moving_avg = np.convolve(history, np.ones(window)/window, mode='valid')
            self.ax.plot(range(window-1, len(history)), moving_avg, 
                        color='#e74c3c', linestyle='--', linewidth=1, label='Tendencia')

        self.ax.set_yscale('log')
        self.ax.set_title(f"Convergencia: {problem}", fontsize=12)
        self.ax.set_xlabel("Épocas")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, which="both", linestyle='--', alpha=0.4)
        self.ax.legend()
        
        self.fig.tight_layout()
        self.canvas.draw()

    def _format_config_text(self, config, problem):
        s = f"Problem: {problem}\n"
        s += f"LR: {config.get('LEARNING_RATE')}\n"
        if 'MODEL_CONFIG' in config:
            mc = config['MODEL_CONFIG']
            s += f"Layers: {mc.get('num_layers')}\n"
            s += f"Neurons: {mc.get('hidden_dim')}\n"
            s += f"Activ: {mc.get('activation')}\n"
        if 'PHYSICS_CONFIG' in config:
            s += "\nPhysics:\n"
            for k, v in config['PHYSICS_CONFIG'].items():
                if isinstance(v, (int, float)):
                    s += f"  {k}: {v}\n"
        return s

    # --- MÉTODOS DE EXPORTACIÓN ---

    def save_plot(self):
        """Guarda la gráfica actual como imagen."""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf")],
                title="Guardar Gráfica de Entrenamiento"
            )
            if file_path:
                self.fig.savefig(file_path, dpi=300)
                messagebox.showinfo("Éxito", f"Gráfica guardada en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la imagen:\n{e}")

    def save_data(self):
        """Guarda el historial de pérdida en un CSV."""
        trainer = self.shared_state.get('trainer')
        if not trainer: return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV File", "*.csv"), ("Excel File", "*.xlsx")],
                title="Guardar Datos de Entrenamiento"
            )
            if file_path:
                # Crear DataFrame
                history = trainer.loss_history
                df = pd.DataFrame({
                    'epoch': range(1, len(history) + 1),
                    'total_loss': history
                })
                
                # Guardar
                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)
                    
                messagebox.showinfo("Éxito", f"Datos guardados en:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron guardar los datos:\n{e}")

    def save_model(self):
        """Guarda el modelo de Keras (.keras o .h5)."""
        trainer = self.shared_state.get('trainer')
        if not trainer or not trainer.model: return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".keras",
                filetypes=[("Keras Model", "*.keras"), ("HDF5 Model", "*.h5")],
                title="Guardar Modelo Entrenado"
            )
            if file_path:
                # Guardar usando la API nativa de Keras
                trainer.model.save(file_path)
                
                # Opcional: Guardar configuración JSON al lado para recordar parámetros físicos
                config_path = file_path + ".config.json"
                with open(config_path, 'w') as f:
                    # Convertir valores numpy a tipos nativos de python para JSON
                    clean_config = self._make_serializable(trainer.config)
                    json.dump(clean_config, f, indent=4)

                messagebox.showinfo("Éxito", f"Modelo guardado exitosamente.\nConfiguración guardada en .json adjunto.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el modelo:\n{e}")

    def _make_serializable(self, config):
        """Convierte objetos no serializables (como float32) a nativos."""
        import copy
        cfg = copy.deepcopy(config)
        
        def convert(item):
            if isinstance(item, dict):
                return {k: convert(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [convert(v) for v in item]
            elif hasattr(item, 'item'): # Numpy types
                return item.item()
            return item
            
        return convert(cfg)

    def on_shared_state_change(self, key, value):
        pass