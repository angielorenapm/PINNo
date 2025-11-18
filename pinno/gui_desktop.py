"""
Interfaz gráfica de escritorio (Tkinter) para PINN Interactive Trainer.

Este módulo define la ventana principal de la aplicación y orquesta la comunicación
entre las diferentes pestañas (Educación, Datos, Entrenamiento, Reportes) mediante
un diccionario de estado compartido.

Usage:
    Este script está diseñado para ejecutarse como un módulo dentro del paquete:
    $ python -m pinno.gui_desktop
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf

# Importar módulos del paquete
from .gui_modules.data_explorer import DataExplorer
from .gui_modules.training_tab import TrainingTab
from .gui_modules.report_tab import ReportTab
from .gui_modules.info_tab import InfoTab
from .training import PINNTrainer


class PINNGUI:
    """
    Controlador principal de la interfaz gráfica de usuario.

    Gestiona el ciclo de vida de la aplicación, la inicialización de componentes
    visuales y mantiene el estado global compartido entre las pestañas.

    Attributes:
        root (tk.Tk): Ventana raíz de Tkinter.
        shared_state (dict): Diccionario mutable que actúa como "bus de datos" entre pestañas.
            Contiene:
            - 'trainer': Instancia de PINNTrainer (o None).
            - 'is_training': Booleano de control de hilos.
            - 'problem_name': tk.StringVar con el problema seleccionado.
            - 'current_dataframe': Datos externos cargados (Pandas).
            - 'loss_history': Lista con el historial de pérdidas.
    """

    def __init__(self, root: tk.Tk):
        """
        Inicializa la aplicación GUI.

        Args:
            root (tk.Tk): Instancia raíz de la ventana de Tkinter.
        """
        self.root = root
        self.root.title("PINN Interactive Trainer v2.0")
        self.root.geometry("1200x750")
        
        # Estado compartido: Fuente de la verdad para todas las pestañas
        self.shared_state = {
            'trainer': None,
            'is_training': False,
            'problem_name': tk.StringVar(value="SHO"),
            'current_dataframe': None,
            'external_data_path': None, 
            'loss_history': [],
            'epoch': 0
        }
        
        self._initialize_ui()
        # Capturar evento de cierre para detener hilos seguramente
        self.root.protocol("WM_DELETE_WINDOW", self._safe_shutdown)

    def _initialize_ui(self):
        """
        Construye y empaqueta los componentes visuales principales.
        
        Crea el widget Notebook (pestañas) e instancia cada módulo (Info, Data,
        Train, Report), inyectando el `shared_state` en cada uno.
        """
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Pestaña Educativa
        self.info_tab = InfoTab(self.notebook, self.shared_state)
        self.notebook.add(self.info_tab, text="📚 Learn PINNs")

        # 2. Explorador de Datos
        self.data_explorer = DataExplorer(self.notebook, self.shared_state)
        self.notebook.add(self.data_explorer, text="📂 Data Exploration")
        
        # 3. Entrenamiento y Configuración
        self.training_tab = TrainingTab(self.notebook, self.shared_state)
        self.notebook.add(self.training_tab, text="⚙️ Train & Config")
        
        # 4. Reporte de Métricas
        self.report_tab = ReportTab(self.notebook, self.shared_state)
        self.notebook.add(self.report_tab, text="📊 Metrics Report")
        
        # Establecer referencias cruzadas para comunicación directa si es necesaria
        self.training_tab.report_tab_ref = self.report_tab
        self.report_tab.training_tab_ref = self.training_tab
        
        self._create_status_bar()

    def _create_status_bar(self):
        """Crea la barra de estado inferior para mensajes del sistema."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        self.status_var = tk.StringVar(value="Ready - Go to 'Learn PINNs' to start.")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(fill=tk.X)

    def _safe_shutdown(self):
        """
        Maneja el cierre de la aplicación.
        
        Asegura que la bandera 'is_training' sea False para detener cualquier
        hilo de entrenamiento en ejecución antes de destruir la ventana.
        """
        self.shared_state['is_training'] = False
        self.root.destroy()

    def run(self):
        """Inicia el bucle principal de eventos de Tkinter."""
        self.root.mainloop()


def launch_gui() -> PINNGUI:
    """
    Configura el entorno (semillas aleatorias) e instancia la GUI.

    Returns:
        PINNGUI: La instancia de la aplicación creada.
    """
    np.random.seed(42)
    tf.random.set_seed(42)
    root = tk.Tk()
    app = PINNGUI(root)
    return app


def main():
    """
    Punto de entrada principal para la ejecución del script.
    """
    app = launch_gui()
    app.run()


if __name__ == "__main__":
    main()