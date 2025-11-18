"""
Módulo para la definición de arquitecturas de redes neuronales usando TensorFlow/Keras.

Este script actúa como una fábrica de modelos, conteniendo las clases que definen
la topología de las redes neuronales (ej. Perceptrón Multicapa) utilizadas para
aproximar la solución de las Ecuaciones Diferenciales Parciales (EDP).

Design:
    El módulo está desacoplado de la física del problema; solo define grafos de
    computación (Input -> Hidden Layers -> Output).
"""

import tensorflow as tf
from typing import Dict, Any

# ==============================================================================
# --- DEFINICIÓN DE ARQUITECTURAS ---
# ==============================================================================

class PINN_MLP(tf.keras.Model):
    """
    Red Neuronal Densa de Múltiples Capas (MLP) para una PINN.
    
    Implementa una arquitectura Feed-Forward (hacia adelante) estándar utilizando
    la API de subclases de Keras. Es la arquitectura base más común para resolver
    EDPs simples.

    Attributes:
        network (tf.keras.Sequential): El contenedor secuencial de capas Keras
            que define el grafo de operaciones.
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
            input_dim (int): Dimensión del espacio de entrada (ej. 1 para t, 3 para x,y,t).
            output_dim (int): Dimensión de la salida (ej. 1 para u(x,t)).
            hidden_dim (int): Número de neuronas en cada capa oculta.
            num_layers (int): Número total de capas (incluyendo entrada y salida).
                              Debe ser al menos 2.
            activation (str, optional): Función de activación no lineal para las capas ocultas.
                                        Se recomienda "tanh" para problemas físicos por su
                                        suavidad en las derivadas. Defaults to "tanh".

        Raises:
            ValueError: Si `num_layers` es menor a 2.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("El número de capas debe ser al menos 2 (entrada y salida).")

        # Construcción dinámica de la lista de capas
        layers = []
        
        # 1. Capa de entrada (implícita en la primera densa o explícita)
        # Usamos input_shape en la primera capa para definir la entrada
        layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation, input_shape=(input_dim,)))
        
        # 2. Capas ocultas intermedias
        # Restamos 2 porque ya añadimos la primera oculta (conectada a input) y falta la de salida
        for _ in range(num_layers - 2):
            layers.append(tf.keras.layers.Dense(hidden_dim, activation=activation))
            
        # 3. Capa de salida (Lineal, sin activación usualmente para regresión)
        layers.append(tf.keras.layers.Dense(output_dim))
        
        self.network = tf.keras.Sequential(layers)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Realiza el paso hacia adelante (Forward Pass) de la red.

        Args:
            x (tf.Tensor): Tensor de entrada con forma (batch_size, input_dim).

        Returns:
            tf.Tensor: Tensor de salida con la aproximación de la solución, 
                       con forma (batch_size, output_dim).
        """
        return self.network(x)


# ==============================================================================
# --- FÁBRICA DE MODELOS (FACTORY PATTERN) ---
# ==============================================================================

MODELS: Dict[str, type] = {
    "mlp": PINN_MLP
}
"""dict: Registro de arquitecturas disponibles mapeadas por nombre."""

def get_model(model_name: str, model_config: Dict[str, Any]) -> tf.keras.Model:
    """
    Instancia una red neuronal a partir de su nombre y configuración.

    Esta función actúa como un despachador (Factory) que simplifica la creación
    de modelos sin necesidad de importar las clases específicas en el código cliente.

    Args:
        model_name (str): Identificador del modelo (ej. "mlp"). Insensible a mayúsculas.
        model_config (Dict[str, Any]): Diccionario de hiperparámetros que se pasarán
                                       como **kwargs al constructor del modelo.
                                       Ej: {'input_dim': 1, 'hidden_dim': 64...}

    Returns:
        tf.keras.Model: Una instancia compilada (o lista para compilar) del modelo Keras.

    Raises:
        ValueError: Si `model_name` no se encuentra en el registro `MODELS`.
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        valid_options = list(MODELS.keys())
        raise ValueError(f"Modelo '{model_name}' no reconocido. Opciones: {valid_options}")
    
    model_class = MODELS[model_name]
    # Desempaquetar configuración directamente en el constructor
    return model_class(**model_config)