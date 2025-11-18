"""
Interfaz gráfica principal para PINN Interactive Trainer (Modificada)
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf

# Importar módulos
from gui_modules.data_explorer import DataExplorer
from gui_modules.training_tab import TrainingTab
from gui_modules.report_tab import ReportTab
from gui_modules.info_tab import InfoTab  # NUEVO

class PINNGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PINN Interactive Trainer v2.0")
        self.root.geometry("1200x750")
        
        self.shared_state = {
            'trainer': None,
            'is_training': False,
            'problem_name': tk.StringVar(value="SHO"),
            'current_dataframe': None,
            'external_data_path': None, # Nuevo para guardar ruta
            'loss_history': [],
            'epoch': 0
        }
        
        self._initialize_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._safe_shutdown)

    def _initialize_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Info Tab (Nuevo)
        self.info_tab = InfoTab(self.notebook, self.shared_state)
        self.notebook.add(self.info_tab, text="📚 Learn PINNs")

        # 2. Data Explorer
        self.data_explorer = DataExplorer(self.notebook, self.shared_state)
        self.notebook.add(self.data_explorer, text="📂 Data Exploration")
        
        # 3. Train & Visualize
        self.training_tab = TrainingTab(self.notebook, self.shared_state)
        self.notebook.add(self.training_tab, text="⚙️ Train & Config")
        
        # 4. Metrics Report
        self.report_tab = ReportTab(self.notebook, self.shared_state)
        self.notebook.add(self.report_tab, text="📊 Metrics Report")
        
        # Referencias cruzadas
        self.training_tab.report_tab_ref = self.report_tab
        self.report_tab.training_tab_ref = self.training_tab
        
        self._create_status_bar()

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        self.status_var = tk.StringVar(value="Ready - Go to 'Learn PINNs' to start.")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)

    def _safe_shutdown(self):
        self.shared_state['is_training'] = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def launch_gui():
    np.random.seed(42)
    tf.random.set_seed(42)
    root = tk.Tk()
    app = PINNGUI(root)
    return app

if __name__ == "__main__":
    app = launch_gui()
    app.run()