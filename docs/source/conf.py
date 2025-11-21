# docs/source/conf.py
import os
import sys

# --- 1. RUTAS ---
# Apuntar a la raíz del proyecto para encontrar la carpeta 'pinno'
sys.path.insert(0, os.path.abspath('../..'))

# --- 2. PROYECTO ---
project = 'PINNo Solver'
copyright = '2023, Universidad Distrital Francisco José de Caldas'
author = 'Angie Pineda, Juan Acuña, Pablo Patiño'
release = '1.0.0'

# --- 3. EXTENSIONES ---
extensions = [
    'sphinx.ext.autodoc',      # Extrae docstrings automáticamente
    'sphinx.ext.napoleon',     # Soporta estilo Google/NumPy en docstrings
    'sphinx.ext.viewcode',     # Agrega enlaces al código fuente
    'sphinx.ext.githubpages',  # Para publicar en GitHub Pages (opcional)
]

# Configuración de Napoleón (opcional, para que se vea bonito)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# --- MOCKING (Para evitar errores de GUI al documentar) ---
autodoc_mock_imports = ["tkinter", "PIL", "matplotlib.backends.backend_tkagg"]

# --- 4. TEMA ---
html_theme = 'renku' # Tema profesional "Read The Docs"
html_static_path = ['_static']