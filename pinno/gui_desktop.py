# pinno/gui_desktop.py
import tkinter as tk
from tkinter import ttk
import sys
import os

# --- SILENCIAR LOGS DE TENSORFLOW ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mensaje de inicio inmediato en terminal
print("Starting PINNo...")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Importar modulos del paquete
from .gui_modules.data_explorer import DataExplorer
from .gui_modules.training_tab import TrainingTab
from .gui_modules.report_tab import ReportTab
from .gui_modules.info_tab import InfoTab

class PINNGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        
        # 1. Ocultar ventana principal inmediatamente
        self.root.withdraw()
        
        self.root.title("PINNo")
        self.root.geometry("1200x750")
        
        # Estado compartido
        self.shared_state = {
            'trainer': None,
            'is_training': False,
            'problem_name': tk.StringVar(value="SHO"),
            'current_dataframe': None,
            'column_mapping': None,
            'time_col': None,
            'use_csv_mode': tk.BooleanVar(value=False),
            'loss_history': [],
            'epoch': 0
        }
        
        # 2. Mostrar Splash Screen
        self._show_splash()
        
        # 3. Cargar UI
        self._initialize_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._safe_shutdown)

    def _show_splash(self):
        """Muestra el logo inicial antes de cargar la app."""
        splash = tk.Toplevel(self.root)
        splash.overrideredirect(True) # Quitar bordes
        
        logo_path = "loadlogo.png"
        
        try:
            pil_image = Image.open(logo_path)
            
            # Ajustar tamano (ancho maximo 600px)
            base_width = 600
            w_percent = (base_width / float(pil_image.size[0]))
            h_size = int((float(pil_image.size[1]) * float(w_percent)))
            
            pil_image = pil_image.resize((base_width, h_size), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Centrar en pantalla
            ws = self.root.winfo_screenwidth()
            hs = self.root.winfo_screenheight()
            x = (ws/2) - (base_width/2)
            y = (hs/2) - (h_size/2)
            
            splash.geometry('%dx%d+%d+%d' % (base_width, h_size, x, y))
            
            label = tk.Label(splash, image=tk_image, bd=0)
            label.pack()
            
            splash.image = tk_image # Referencia
            
        except Exception as e:
            print(f"Splash error (saltando): {e}")
            splash.destroy()
            self.root.deiconify()
            return

        def close_splash():
            splash.destroy()
            self.root.deiconify() # Mostrar app principal
            
        # Duracion: 3 segundos
        self.root.after(3000, close_splash)

    def _initialize_ui(self):
        """Construye la interfaz grafica principal."""
        
        # --- NOTEBOOK (PESTAÑAS) ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 5))
        
        # --- LOGO EN ESQUINA (TASKBAR + HEADER FLOTANTE) ---
        try:
            icon_path = "Logosmall.png"
            img_source = Image.open(icon_path)
            
            # 1. Icono de Ventana / Taskbar
            icon_photo = ImageTk.PhotoImage(img_source)
            self.root.iconphoto(True, icon_photo)
            
            # 2. Logo en Esquina Derecha (Mismo nivel que pestañas)
            # Altura pequena (30px) para igualar la altura de la pestaña
            target_h = 30
            aspect_ratio = img_source.width / img_source.height
            target_w = int(target_h * aspect_ratio)
            
            img_resized = img_source.resize((target_w, target_h), Image.Resampling.LANCZOS)
            self.corner_logo = ImageTk.PhotoImage(img_resized)
            
            # Usamos Label y place() para flotarlo en la esquina superior derecha
            lbl_logo = tk.Label(self.root, image=self.corner_logo, bd=0)
            # relx=1.0 (Borde derecho), y=0 (Arriba), anchor='ne' (Esquina NorEste del label)
            # x=-10 da un pequeño margen desde el borde derecho
            lbl_logo.place(relx=1.0, x=-15, y=2, anchor="ne")
            
        except Exception as e:
            print(f"Warning: No se pudo cargar Logosmall.png para la esquina: {e}")

        # --- PESTAÑAS ---
        self.info_tab = InfoTab(self.notebook, self.shared_state)
        self.notebook.add(self.info_tab, text="Learn PINNs")

        self.data_explorer = DataExplorer(self.notebook, self.shared_state)
        self.notebook.add(self.data_explorer, text="Data Exploration")
        
        self.training_tab = TrainingTab(self.notebook, self.shared_state)
        self.notebook.add(self.training_tab, text="Train & Config")
        
        self.report_tab = ReportTab(self.notebook, self.shared_state)
        self.notebook.add(self.report_tab, text="Metrics Report")
        
        self.training_tab.report_tab_ref = self.report_tab
        self.report_tab.training_tab_ref = self.training_tab
        
        self._create_status_bar()

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        self.status_var = tk.StringVar(value="Ready - Go to 'Learn PINNs' to start.")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)

    def _safe_shutdown(self):
        """Cierre limpio usando os._exit para matar procesos de TF."""
        print("PINNo closing...")
        self.shared_state['is_training'] = False
        try:
            self.root.destroy()
        except:
            pass
        os._exit(0)

    def run(self):
        self.root.mainloop()

def launch_gui() -> PINNGUI:
    # Configurar semillas
    np.random.seed(42)
    tf.random.set_seed(42)
    
    root = tk.Tk()
    
    # Tema visual opcional
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    app = PINNGUI(root)
    return app

def main():
    app = launch_gui()
    app.run()

if __name__ == "__main__":
    main()
