import sys
import os

# Aseguramos que Python encuentre el paquete actual
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Importamos usando el nuevo nombre de carpeta 'pinno'
from pinno.gui_desktop import main

if __name__ == "__main__":
    main()