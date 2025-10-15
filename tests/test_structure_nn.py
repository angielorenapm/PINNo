# tests/structure_nn_test.py
import sys
import os
import tensorflow as tf

# --- Bloque de código para encontrar la carpeta 'src' ---
# 1. Obtiene la ruta del directorio donde está este archivo (la carpeta 'tests')
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Obtiene la ruta del directorio padre (la raíz del proyecto 'PINNo/')
project_root = os.path.dirname(current_dir)
# 3. Añade la raíz del proyecto al path de Python para que encuentre 'src'
sys.path.insert(0, project_root)
# --- Fin del bloque ---

# Ahora la importación de 'src' funcionará sin problemas
from src.models import get_model, PINN_MLP

# --- Tu Lógica de Prueba ---
# (Puedes pegar tus funciones de prueba aquí)

def test_model_creation_successfully():
    """Verifica que la fábrica 'get_model' cree un modelo correctamente."""
    print("Probando creación de modelo...")
    config = {
        "input_dim": 2, "output_dim": 1, "hidden_dim": 32,
        "num_layers": 3, "activation": "tanh"
    }
    model = get_model("mlp", config)
    assert isinstance(model, PINN_MLP)
    print("-> Creación exitosa.")

def test_forward_pass_shape():
    """Verifica que la forma de la salida del modelo sea la correcta."""
    print("Probando 'forward pass'...")
    config = {
        "input_dim": 2, "output_dim": 1, "hidden_dim": 32,
        "num_layers": 3, "activation": "tanh"
    }
    model = get_model("mlp", config)
    dummy_input = tf.random.normal(shape=(10, 2))
    output = model(dummy_input)
    assert output.shape == (10, 1)
    print("-> Forma de salida correcta.")


# --- Punto de Entrada para Ejecutar el Script ---
# Este bloque se asegura de que las pruebas se ejecuten cuando corres el archivo
if __name__ == "__main__":
    print("🧪 Ejecutando pruebas manualmente...")
    test_model_creation_successfully()
    test_forward_pass_shape()
    print("\n✅ Todas las pruebas manuales finalizaron.")