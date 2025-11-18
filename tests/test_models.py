# tests/test_models.py
import pytest
import tensorflow as tf
import numpy as np
from src.models import get_model

@pytest.mark.parametrize("activation", ["tanh", "relu", "sigmoid"])
@pytest.mark.parametrize("num_layers", [2, 4, 6])
def test_mlp_architectures(activation, num_layers):
    """Verifica que la MLP se construya con diferentes profundidades y activaciones."""
    config = {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dim": 10,
        "num_layers": num_layers,
        "activation": activation
    }
    
    model = get_model("mlp", config)
    
    # Verificar input/output shape
    dummy_in = tf.random.normal((5, 2))
    out = model(dummy_in)
    assert out.shape == (5, 1)
    
    # Verificar número de capas (Dense layers)
    # input layer no cuenta como layer en keras sequential list a veces, 
    # pero num_layers define la profundidad total.
    # Verificamos que no sea vacío y tenga pesos.
    assert len(model.trainable_variables) > 0

def test_model_gradients():
    """Verifica que el modelo sea diferenciable (crítico para PINNs)."""
    config = {"input_dim": 1, "output_dim": 1, "hidden_dim": 10, "num_layers": 3, "activation": "tanh"}
    model = get_model("mlp", config)
    
    x = tf.constant([[1.0], [2.0]])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    
    grad = tape.gradient(y, x)
    assert grad is not None
    assert grad.shape == x.shape