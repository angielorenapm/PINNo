import sys
import os
from streamlit.web import cli as stcli

def run():
    # Obtener ruta base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Apuntar a la carpeta 'pinno'
    app_path = os.path.join(base_dir, "pinno", "gui_web.py")

    if not os.path.exists(app_path):
        print(f"Error: No se encuentra {app_path}")
        return

    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    run()