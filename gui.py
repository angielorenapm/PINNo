# gui.py
"""
Interfaz gráfica principal para PINN Interactive Trainer

Punto de entrada único para la interfaz gráfica. Coordina las tres pestañas principales:
- Data Exploration: Exploración y visualización de datos CSV
- Train & Visualize: Entrenamiento y monitoreo en tiempo real  
- Metrics Report: Reportes detallados y análisis de resultados

Mantiene un estado global compartido entre todas las pestañas.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf

# Importar módulos de la interfaz
from gui_modules.data_explorer import DataExplorer
from gui_modules.training_tab import TrainingTab
from gui_modules.report_tab import ReportTab


class PINNGUI:
    """
    Ventana principal de la aplicación PINN.
    
    Responsabilidades:
    - Crear y gestionar el notebook con pestañas
    - Mantener estado global compartido entre pestañas
    - Coordinar comunicación entre componentes
    - Gestionar ciclo de vida de la aplicación
    """
    
    def __init__(self, root):
        """
        Inicializa la interfaz gráfica principal.
        
        Args:
            root: Ventana raíz de Tkinter
        """
        self.root = root
        self.root.title("PINN Interactive Trainer")
        self.root.geometry("1200x700")
        
        # Configurar el estado global compartido entre pestañas
        self.shared_state = {
            # Sistema de entrenamiento
            'trainer': None,
            'is_training': False,
            'problem_name': tk.StringVar(value="SHO"),
            
            # Datos y resultados
            'current_dataframe': None,
            'loss_history': [],
            'epoch': 0,
            'current_predictions': None,
            'current_analytical': None,
            
            # Configuración visual
            'plot_domain_limits': None
        }
        
        # Inicializar la interfaz
        self._initialize_ui()
        
        # Configurar cierre seguro
        self.root.protocol("WM_DELETE_WINDOW", self._safe_shutdown)

    def _initialize_ui(self):
        """Inicializa todos los componentes de la interfaz de usuario."""
        # Crear notebook (pestañas) principal
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Inicializar las pestañas
        self._initialize_tabs()
        
        # Barra de estado (opcional)
        self._create_status_bar()

    def _initialize_tabs(self):
        """Inicializa y conecta las tres pestañas principales."""
        # Pestaña 1: Exploración de datos
        self.data_explorer = DataExplorer(self.notebook, self.shared_state)
        self.notebook.add(self.data_explorer, text="Data Exploration")
        
        # Pestaña 2: Entrenamiento y visualización
        self.training_tab = TrainingTab(self.notebook, self.shared_state)
        self.notebook.add(self.training_tab, text="Train & Visualize")
        
        # Pestaña 3: Reportes y métricas
        self.report_tab = ReportTab(self.notebook, self.shared_state)
        self.notebook.add(self.report_tab, text="Metrics Report")
        
        # CONEXIÓN NUEVA: Dar referencias cruzadas para comunicación directa
        self.training_tab.report_tab_ref = self.report_tab
        self.report_tab.training_tab_ref = self.training_tab

    def _create_status_bar(self):
        """Crea una barra de estado en la parte inferior."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.status_var = tk.StringVar(value="Ready - Load data or start training")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X)

    def update_status(self, message: str):
        """
        Actualiza el mensaje de la barra de estado.
        
        Args:
            message: Nuevo mensaje para mostrar
        """
        self.status_var.set(message)
        self.root.update_idletasks()

    def get_shared_state(self, key: str):
        """
        Obtiene un valor del estado global compartido.
        
        Args:
            key: Clave del estado a obtener
            
        Returns:
            Valor almacenado en la clave, o None si no existe
        """
        return self.shared_state.get(key)

    def set_shared_state(self, key: str, value, notify_tabs: bool = True):
        """
        Establece un valor en el estado global compartido.
        
        Args:
            key: Clave del estado a establecer
            value: Valor a almacenar
            notify_tabs: Si es True, notifica a las pestañas del cambio
        """
        self.shared_state[key] = value
        
        if notify_tabs:
            self._notify_tabs(key, value)

    def _notify_tabs(self, key: str, value):
        """
        Notifica a todas las pestañas sobre un cambio en el estado global.
        
        Args:
            key: Clave del estado que cambió
            value: Nuevo valor
        """
        tabs = [self.data_explorer, self.training_tab, self.report_tab]
        
        for tab in tabs:
            if hasattr(tab, 'on_shared_state_change'):
                try:
                    tab.on_shared_state_change(key, value)
                except Exception as e:
                    print(f"Error notifying tab {tab.__class__.__name__}: {e}")

    def _safe_shutdown(self):
        """Cierra la aplicación de manera segura, deteniendo cualquier entrenamiento en curso."""
        if self.shared_state.get('is_training', False):
            # Detener el entrenamiento antes de cerrar
            self.shared_state['is_training'] = False
            self.update_status("Stopping training before shutdown...")
            
            # Esperar un momento para que el entrenamiento se detenga
            self.root.after(100, self._force_shutdown)
        else:
            self._force_shutdown()

    def _force_shutdown(self):
        """Fuerza el cierre de la aplicación."""
        self.root.quit()
        self.root.destroy()

    def run(self):
        """
        Inicia el bucle principal de la aplicación.
        
        Esta función bloquea hasta que la aplicación se cierra.
        """
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._safe_shutdown()
        except Exception as e:
            print(f"Unexpected error in GUI: {e}")
            self._safe_shutdown()


def launch_gui():
    """
    Función de fábrica para crear y lanzar la aplicación GUI.
    
    Returns:
        Instancia de PINNGUI configurada y lista para ejecutar
    """
    # Configurar semillas para reproducibilidad
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Crear ventana raíz
    root = tk.Tk()
    
    # Configuración adicional de la ventana
    root.minsize(1000, 600)
    
    # Crear aplicación
    app = PINNGUI(root)
    
    # Mensaje de inicio
    app.update_status("PINNo started successfully")
    
    return app


# Punto de entrada cuando se ejecuta directamente
if __name__ == "__main__":
    print("Starting PINNo GUI...")
    
    # Lanzar aplicación
    app = launch_gui()
    
    # Ejecutar bucle principal
    app.run()
    
    print("PINNo closed")