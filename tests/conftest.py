# tests/conftest.py
import pytest
import sys
import os
import tensorflow as tf
import numpy as np
from src.config import get_active_config


# Agregar el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_active_config

@pytest.fixture(scope="session")
def sho_config():
    """Configuración SHO reducida para tests rápidos."""
    conf = get_active_config("SHO")
    conf["DATA_CONFIG"]["n_collocation"] = 50
    conf["DATA_CONFIG"]["n_initial"] = 5
    return conf

@pytest.fixture(scope="session")
def heat_config():
    """Configuración HEAT reducida para tests rápidos."""
    conf = get_active_config("HEAT")
    conf["DATA_CONFIG"]["n_collocation"] = 50
    conf["DATA_CONFIG"]["n_initial"] = 10
    conf["DATA_CONFIG"]["n_boundary"] = 10
    return conf

@pytest.fixture
def mock_data_sho(sho_config):
    """Genera tensores dummy para SHO sin llamar al DataManager."""
    t = tf.random.normal((10, 1))
    return {
        "t_coll": t,
        "t0": tf.zeros((1, 1)),
        "x0_true": tf.constant(1.0),
        "v0_true": tf.constant(0.0)
    }
