# tests/test_data_losses.py
import pytest
import tensorflow as tf
import pandas as pd
import os
from unittest.mock import MagicMock, patch
from src.data_manage import DataManager
from src.losses import LossCalculator

def test_data_manager_shapes(sho_config):
    """Verifica que DataManager devuelve tensores con formas correctas."""
    dm = DataManager(sho_config, "SHO")
    dm.prepare_data()
    data = dm.get_training_data()
    
    n_coll = sho_config["DATA_CONFIG"]["n_collocation"]
    assert data["t_coll"].shape == (n_coll, 1)
    assert data["t0"].shape == (1, 1)

def test_csv_loading_integration(tmp_path, sho_config):
    """
    Crea un CSV temporal real y prueba cargarlo.
    Usa 'tmp_path' fixture de pytest para archivos temporales.
    """
    # 1. Crear CSV dummy
    df = pd.DataFrame({
        't': [0.1, 0.2, 0.3],
        'x': [1.0, 0.9, 0.8]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    # 2. Cargar en DataManager
    dm = DataManager(sho_config, "SHO")
    success = dm.load_external_data(str(csv_path))
    dm.prepare_data()
    data = dm.get_training_data()
    
    # 3. Validar
    assert success is True
    assert "ext_t" in data
    assert "ext_x" in data
    assert data["ext_t"].shape == (3, 1)

def test_loss_weights_impact(sho_config, mock_data_sho):
    """Verifica que cambiar los pesos afecta la pérdida final."""
    from src.physics import get_physics_problem
    from src.models import get_model
    
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
    assert loss_b != loss_a