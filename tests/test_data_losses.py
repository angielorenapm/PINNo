# tests/test_data_losses.py
import pytest
import tensorflow as tf
import pandas as pd
import os
import sys
from unittest.mock import MagicMock, patch

# --- PATH PATCHING (CRÍTICO) ---
# Asegura que Python encuentre el paquete 'src' o 'pinno'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTACIONES ROBUSTAS ---
try:
    from pinno.data_manage import DataManager
    from pinno.losses import LossCalculator
    from pinno.models import get_model
    from pinno.physics import get_physics_problem
except ImportError:
    from src.data_manage import DataManager
    from src.losses import LossCalculator
    from src.models import get_model
    from src.physics import get_physics_problem

def test_data_manager_shapes(sho_config):
    """Verifica que DataManager devuelve tensores con formas correctas."""
    dm = DataManager(sho_config, "SHO")
    dm.prepare_data()
    data = dm.get_training_data()
    
    n_coll = sho_config["DATA_CONFIG"]["n_collocation"]
    assert data["t_coll"].shape == (n_coll, 1)
    assert data["t0"].shape == (1, 1)

def test_loss_weights_impact(sho_config, mock_data_sho):
    """Verifica que cambiar los pesos afecta la pérdida final."""
    model = get_model("mlp", sho_config["MODEL_CONFIG"])
    physics = get_physics_problem("SHO", sho_config)
    
    # Caso A: Peso ODE bajo
    weights_a = {"ode": 0.0, "initial": 1.0, "data": 0.0}
    calc_a = LossCalculator(weights_a, "SHO")
    loss_a, _ = calc_a.compute_losses(model, physics, mock_data_sho)
    
    # Caso B: Peso ODE alto
    weights_b = {"ode": 1000.0, "initial": 1.0, "data": 0.0}
    calc_b = LossCalculator(weights_b, "SHO")
    loss_b, _ = calc_b.compute_losses(model, physics, mock_data_sho)
    
    # Es altamente probable que loss_b sea mayor que loss_a si el modelo no está entrenado
    # Nota: loss_a solo considera initial condition, loss_b considera ODE con peso 1000
    assert loss_b != loss_a
