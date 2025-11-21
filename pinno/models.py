# pinno/models.py
import tensorflow as tf
from typing import Dict, Any

class PINN_MLP(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "tanh"
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("El numero de capas debe ser al menos 2.")

        layers = []
        
        # CORRECCION: Usar InputLayer explícito
        layers.append(tf.keras.layers.InputLayer(shape=(input_dim,)))
        
        # Primera capa oculta
        layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))
        
        # Capas ocultas intermedias
        for _ in range(num_layers - 2):
            layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))
        
        # Capa de salida
        layers.append(tf.keras.layers.Dense(output_dim))
        
        self.network = tf.keras.Sequential(layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.network(x)

MODELS = {"mlp": PINN_MLP}

def get_model(model_name: str, model_config: Dict[str, Any]) -> tf.keras.Model:
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")
    return MODELS[model_name](**model_config)