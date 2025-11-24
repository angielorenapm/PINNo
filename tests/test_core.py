# tests/test_core.py
import unittest
import numpy as np
import tensorflow as tf
import sys
import os
from unittest.mock import patch, MagicMock

# --- PATH PATCHING (CRÍTICO) ---
# Asegura que Python encuentre el paquete 'src' o 'pinno'
# subiendo un nivel desde la carpeta 'tests'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTACIONES ROBUSTAS ---
# Intenta importar desde 'pinno' (nombre nuevo) o 'src' (nombre antiguo)
try:
    from pinno.models import get_model
    from pinno.physics import get_physics_problem
    from pinno.data_manage import DataManager
    from pinno.losses import LossCalculator
    from pinno.config import get_active_config
except ImportError:
    from src.models import get_model
    from src.physics import get_physics_problem
    from src.data_manage import DataManager
    from src.losses import LossCalculator
    from src.config import get_active_config

class TestPINNComponents(unittest.TestCase):
    
    def setUp(self):
        """Configuración previa para cada test."""
        # Configuración básica para SHO
        self.sho_config = get_active_config("SHO")
        # Configuración básica para HEAT
        self.heat_config = get_active_config("HEAT")
        
        # Reducir epochs y puntos para que los tests sean rápidos
        self.sho_config["DATA_CONFIG"]["n_collocation"] = 10
        self.heat_config["DATA_CONFIG"]["n_collocation"] = 10

    # --- 1. TEST DE MODELOS ---
    def test_model_creation_and_shape(self):
        """Verifica que el modelo se crea y devuelve la forma correcta."""
        model_config = self.sho_config["MODEL_CONFIG"]
        model = get_model("mlp", model_config)
        
        # Crear un input dummy (batch_size=5, input_dim=1)
        dummy_input = tf.random.normal((5, model_config["input_dim"]))
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (5, model_config["output_dim"]))
        print("✅ Model Shape Test Passed")

    # --- 2. TEST DE FÍSICA (SHO) ---
    def test_physics_sho_residual(self):
        """Verifica el cálculo del residual para el Oscilador Armónico."""
        physics = get_physics_problem("SHO", self.sho_config)
        model = get_model("mlp", self.sho_config["MODEL_CONFIG"])
        
        # Input temporal dummy
        t = tf.random.normal((10, 1))
        
        # Calcular residual
        residual = physics.pde_residual(model, t)
        
        # Debe devolver un tensor de (10, 1)
        self.assertEqual(residual.shape, (10, 1))
        print("✅ SHO Physics Residual Test Passed")

    # --- 4. TEST DE GESTIÓN DE DATOS ---
    def test_data_generation_sho(self):
        """Verifica que DataManager genere las claves correctas para SHO."""
        dm = DataManager(self.sho_config, "SHO")
        dm.prepare_data()
        data = dm.get_training_data()
        
        required_keys = ["t_coll", "t0", "x0_true", "v0_true"]
        for key in required_keys:
            self.assertIn(key, data)
            self.assertIsInstance(data[key], tf.Tensor)
        print("✅ SHO Data Generation Test Passed")


    # --- 5. TEST DE PÉRDIDAS (LOSSES) ---
    def test_loss_calculation(self):
        """Verifica que el cálculo de pérdida devuelva un escalar."""
        # Setup completo mini
        model = get_model("mlp", self.sho_config["MODEL_CONFIG"])
        physics = get_physics_problem("SHO", self.sho_config)
        dm = DataManager(self.sho_config, "SHO")
        dm.prepare_data()
        data = dm.get_training_data()
        
        calc = LossCalculator(self.sho_config["LOSS_WEIGHTS"], "SHO")
        
        # Calcular Loss
        total_loss, components = calc.compute_losses(model, physics, data)
        
        # Verificar tipos
        self.assertTrue(tf.is_tensor(total_loss))
        self.assertEqual(len(total_loss.shape), 0) # Debe ser escalar
        self.assertIsInstance(components, list)
        print("✅ Loss Calculation Test Passed")

if __name__ == '__main__':
    unittest.main(verbosity=2)
