# pinno/models.py
"""
Módulo para la definición de arquitecturas de redes neuronales usando TensorFlow/Keras.

Este script actúa como una fábrica de modelos, conteniendo las clases que definen
la topología de las redes neuronales (ej. Perceptrón Multicapa) utilizadas para
aproximar la solución de las ecuaciones diferenciales.
"""

import tensorflow as tf
from typing import Dict, Any

class PINN_MLP(tf.keras.Model):
    """
    Red Neuronal Densa de Múltiples Capas (MLP) para una PINN.
    
    Implementa una arquitectura 'Feed-Forward' estándar. Es la arquitectura base 
    más común para resolver ecuaciones diferenciales simples.
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
        Inicializa la arquitectura de la red neuronal.

        Args:
            input_dim (int): Dimensión de entrada (ej. 1 para t, 3 para x,y,t).
            output_dim (int): Dimensión de salida (ej. 1 para u).
            hidden_dim (int): Número de neuronas por capa oculta.
            num_layers (int): Número total de capas (incluyendo entrada y salida).
            activation (str, optional): Función de activación. Por defecto es "tanh".

        Raises:
            ValueError: Si el número de capas es menor a 2.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("El numero de capas debe ser al menos 2.")

        layers = []
        
        # CORRECCION: Usar InputLayer explícito para evitar warnings de Keras
        layers.append(tf.keras.layers.InputLayer(shape=(input_dim,)))
        
        # Primera capa oculta
        layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))
        
        # Capas ocultas intermedias
        for _ in range(num_layers - 2):
            layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))
        
        # Capa de salida (Lineal)
        layers.append(tf.keras.layers.Dense(output_dim))
        
        self.network = tf.keras.Sequential(layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Realiza el paso hacia adelante (Forward Pass).

        Args:
            x (tf.Tensor): Tensor de entrada.

        Returns:
            tf.Tensor: Tensor de salida con la predicción de la red.
        """
        return self.network(x)

MODELS = {"mlp": PINN_MLP}

def get_model(model_name: str, model_config: Dict[str, Any]) -> tf.keras.Model:
    """
    Fábrica que instancia un modelo a partir de su nombre y configuración.

    Args:
        model_name (str): Identificador del modelo (ej. "mlp").
        model_config (Dict[str, Any]): Diccionario de hiperparámetros para el constructor.

    Raises:
        ValueError: Si el nombre del modelo no está registrado.

    Returns:
        tf.keras.Model: Instancia compilada del modelo solicitado.
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Modelo '{model_name}' no reconocido.")
    return MODELS[model_name](**model_config)