"""
Módulo para la definición de arquitecturas de redes neuronales usando TensorFlow/Keras.

Este script contiene las clases de los modelos (ej. Redes Neuronales de Múltiples Capas)
que se usarán para aproximar la solución de las ecuaciones diferenciales.
Está diseñado para ser independiente de la física del problema o del bucle de entrenamiento.
"""

import tensorflow as tf
from typing import Dict, Any, List

# --- Definición de Arquitecturas de Modelos ---

class PINN_MLP(tf.keras.Model):
    """
    Red Neuronal Densa de Múltiples Capas (MLP) para una PINN.

    Implementación en TensorFlow/Keras de la arquitectura más común para PINNs,
    compuesta por una secuencia de capas densas con funciones de activación.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "tanh"
    ):
        """
        Inicializador del MLP.

        Args:
            input_dim (int): Dimensión del espacio de entrada (ej. 2 para (x, t)).
            output_dim (int): Dimensión del espacio de salida (ej. 1 para u(x,t)).
            hidden_dim (int): Número de neuronas en cada capa oculta.
            num_layers (int): Número total de capas densas en la red.
            activation (str): El nombre de la función de activación a usar
                                entre las capas ocultas (ej. "tanh", "relu").
        """
        super().__init__()

        if num_layers < 2:
            raise ValueError("El número de capas debe ser al menos 2 (entrada y salida).")

        layers: List[tf.keras.layers.Layer] = []
        
        # Capa de entrada (Keras infiere la forma de entrada en la primera llamada,
        # pero podemos ser explícitos para mayor claridad)

        layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation, input_shape=(input_dim,)))

        # Capas ocultas
        for _ in range(num_layers - 2):
            layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))

        # Capa de salida (generalmente con activación lineal, por defecto)
        layers.append(tf.keras.layers.Dense(output_dim))

        # Crear la red secuencial
        self.network = tf.keras.Sequential(layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Define el forward pass de la red (en Keras se llama 'call')."""
        return self.network(x)

# --- Funciones de Ayuda (Fábrica de Modelos) ---

# Mapeo de strings a clases de modelos. Keras maneja los strings de activación directamente.
MODELS: Dict[str, type] = {
    "mlp": PINN_MLP
    # Podrías agregar más arquitecturas aquí, ej: "resnet_mlp": ResNetMLP_TF
}

def get_model(model_name: str, model_config: Dict[str, Any]) -> tf.keras.Model:
    """
    Fábrica de modelos que instancia una red neuronal a partir de su nombre y configuración.

    Args:
        model_name (str): El nombre de la arquitectura a usar (ej. "mlp").
        model_config (Dict[str, Any]): Un diccionario con los parámetros para el
                                        constructor del modelo.

    Returns:
        tf.keras.Model: Una instancia del modelo de Keras solicitado.
        
    Raises:
        ValueError: Si el nombre del modelo no es válido.
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Modelo '{model_name}' no reconocido. Opciones disponibles: {list(MODELS.keys())}")

    model_class = MODELS[model_name]
    
    return model_class(**model_config)