#src/models.py
"""
Módulo para la definición de arquitecturas de redes neuronales usando TensorFlow/Keras.

Este script contiene las clases de los modelos (ej. Redes Neuronales de Múltiples Capas)
que se usarán para aproximar la solución de las ecuaciones diferenciales.
Está diseñado para ser independiente de la física del problema o del bucle de entrenamiento.
"""

import tensorflow as tf
from typing import Dict, Any

# --- Definición de Arquitecturas de Modelos ---

class PINN_MLP(tf.keras.Model):
    """
    Red Neuronal Densa de Múltiples Capas (MLP) para una PINN.
    
    Implementación en TensorFlow/Keras de la arquitectura más común para PINNs.
    """
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
            raise ValueError("El número de capas debe ser al menos 2 (entrada y salida).")

        layers = [tf.keras.layers.Dense(hidden_dim, activation=activation, input_shape=(input_dim,))]
        for _ in range(num_layers - 2):
            layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))
        layers.append(tf.keras.layers.Dense(output_dim))
        self.network = tf.keras.Sequential(layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Define el forward pass de la red (en Keras se llama 'call')."""
        return self.network(x)

# --- Funciones de Ayuda (Fábrica de Modelos) ---

MODELS: Dict[str, type] = {
    "mlp": PINN_MLP
}

def get_model(model_name: str, model_config: Dict[str, Any]) -> tf.keras.Model:
    """
    Fábrica de modelos que instancia una red neuronal a partir de su nombre y configuración.
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Modelo '{model_name}' no reconocido. Opciones: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name]
    return model_class(**model_config)