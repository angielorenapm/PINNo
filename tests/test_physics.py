# tests/test_physics.py
import pytest
import tensorflow as tf
import numpy as np
import sys
import os

# --- PATH PATCHING (CRÍTICO) ---
# Asegura que Python encuentre el paquete 'src' o 'pinno'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTACIONES ROBUSTAS ---
try:
    from pinno.physics import get_physics_problem
    from pinno.models import get_model
    from pinno.config import ALL_CONFIGS
except ImportError:
    from src.physics import get_physics_problem
    from src.models import get_model
    from src.config import ALL_CONFIGS

def test_sho_residual_calculation(sho_config):
    physics = get_physics_problem("SHO", sho_config)
    model = get_model("mlp", sho_config["MODEL_CONFIG"])
    
    t = tf.random.uniform((20, 1))
    residual = physics.pde_residual(model, t)
    
    assert residual.shape == (20, 1)
    assert residual.dtype == tf.float32
