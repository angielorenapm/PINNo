# tests/test_physics.py
import pytest
import tensorflow as tf
import numpy as np
from src.physics import get_physics_problem
from src.models import get_model

def test_sho_residual_calculation(sho_config):
    physics = get_physics_problem("SHO", sho_config)
    model = get_model("mlp", sho_config["MODEL_CONFIG"])
    
    t = tf.random.uniform((20, 1))
    residual = physics.pde_residual(model, t)
    
    assert residual.shape == (20, 1)
    assert residual.dtype == tf.float32

def test_dho_dumping_logic():
    """Verifica que DHO cambie si zeta < 1 (subamortiguado)."""
    from src.config import ALL_CONFIGS
    dho_conf = ALL_CONFIGS["DHO"].copy()
    
    # Caso 1: Subamortiguado
    dho_conf["PHYSICS_CONFIG"]["zeta"] = 0.1
    physics = get_physics_problem("DHO", dho_conf)
    t = tf.constant([[1.0]])
    sol_sub = physics.analytical_solution(t)
    
    # Caso 2: Sobreamortiguado (zeta > 1) -> Según código actual devuelve 0
    dho_conf["PHYSICS_CONFIG"]["zeta"] = 2.0
    physics_over = get_physics_problem("DHO", dho_conf)
    sol_over = physics_over.analytical_solution(t)
    
    assert sol_sub != 0
    assert float(sol_over) == 0.0  # Según la lógica actual en physics.py

def test_heat_slicing_metric(heat_config):
    """Prueba la función de métrica de corte (slicing)."""
    physics = get_physics_problem("HEAT", heat_config)
    model = get_model("mlp", heat_config["MODEL_CONFIG"])
    
    # Probar con t=0 (condición inicial conocida)
    mse, pred, true = physics.compute_slice_metrics(model, t_fix=0.0, resolution=10)
    
    assert isinstance(mse, (float, np.floating))
    assert pred.shape == (10, 10)
    assert true.shape == (10, 10)
    
    # La solución analítica en t=0 no debe ser toda ceros (es sin(pi*x)*sin(pi*y))
    assert np.mean(np.abs(true)) > 0