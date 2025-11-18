# tests/test_core.py
import unittest
import numpy as np
import tensorflow as tf
import sys
import os
from unittest.mock import patch, MagicMock

# Añadir el directorio raíz al path para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    # --- 3. TEST DE FÍSICA (HEAT 2D) ---
    def test_physics_heat_residual_and_slice(self):
        """Verifica residual y función de slicing para Calor 2D."""
        physics = get_physics_problem("HEAT", self.heat_config)
        model = get_model("mlp", self.heat_config["MODEL_CONFIG"])
        
        # Input dummy (x, y, t) -> (batch, 3)
        xyt = tf.random.normal((10, 3))
        
        # 1. Test Residual
        residual = physics.pde_residual(model, xyt)
        self.assertEqual(residual.shape, (10, 1))
        
        # 2. Test Slice Metrics (Nueva funcionalidad)
        mse, u_pred, u_true = physics.compute_slice_metrics(model, t_fix=0.5, resolution=20)
        
        self.assertIsInstance(mse, (float, np.floating))
        self.assertEqual(u_pred.shape, (20, 20)) # Resolución definida arriba
        print("✅ HEAT Physics & Slicing Test Passed")

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

    @patch('src.data_manage.pd.read_csv')
    def test_external_data_loading(self, mock_read_csv):
        """Verifica la carga de datos externos (Mockeando pandas)."""
        # Simular DataFrame de pandas
        mock_df = MagicMock()
        mock_df.columns = ['t', 'x']
        # Simular valores .values.reshape
        mock_val = np.array([[0.1], [0.2]], dtype=np.float32)
        mock_df.__getitem__.return_value.values.reshape.return_value = mock_val
        
        mock_read_csv.return_value = mock_df
        
        dm = DataManager(self.sho_config, "SHO")
        success = dm.load_external_data("dummy_data.csv")
        
        self.assertTrue(success)
        
        # Forzar preparación para inyectar datos
        dm.prepare_data()
        data = dm.get_training_data()
        
        self.assertIn("ext_t", data)
        self.assertIn("ext_x", data)
        print("✅ External Data Loading Test Passed")

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