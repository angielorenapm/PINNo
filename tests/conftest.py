# tests/conftest.py
import pytest
import sys
import os
import tensorflow as tf
import numpy as np

# --- PATH PATCHING PARA TESTS ---
# Obtener la ruta absoluta de la carpeta raíz del proyecto (un nivel arriba de tests/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Insertar la raíz en sys.path si no está. Esto es CRUCIAL para que pytest encuentre
# el paquete 'src' o 'pinno' sin necesidad de instalarlo.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -------------------------------

# Intenta importar usando 'src' o 'pinno' según corresponda
# Esto hace que los tests sean robustos ante el cambio de nombre de la carpeta.
try:
    from src.config import get_active_config
except ImportError:
    from pinno.config import get_active_config

@pytest.fixture(scope="session")
def sho_config():
    """
    Fixture que devuelve una configuración SHO reducida para tests rápidos.
    'scope="session"' significa que se ejecuta una vez por toda la sesión de pruebas.
    """
    conf = get_active_config("SHO")
    # Reducimos la cantidad de puntos para que los tests no tarden una eternidad
    conf["DATA_CONFIG"]["n_collocation"] = 50
    conf["DATA_CONFIG"]["n_initial"] = 5
    return conf

@pytest.fixture(scope="session")
def heat_config():
    """
    Fixture que devuelve una configuración HEAT reducida para tests rápidos.
    """
    conf = get_active_config("HEAT")
    conf["DATA_CONFIG"]["n_collocation"] = 50
    conf["DATA_CONFIG"]["n_initial"] = 10
    conf["DATA_CONFIG"]["n_boundary"] = 10
    return conf

@pytest.fixture
def mock_data_sho(sho_config):
    """
    Genera tensores dummy para SHO sin llamar al DataManager.
    Útil para probar componentes aislados (como Losses) sin depender de la generación de datos.
    """
    # Creamos tensores de TensorFlow directamente
    t = tf.random.normal((10, 1))
    return {
        "t_coll": t,
        "t0": tf.zeros((1, 1)),
        "x0_true": tf.constant(1.0),
        "v0_true": tf.constant(0.0)
    }
