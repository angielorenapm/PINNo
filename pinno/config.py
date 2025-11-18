"""
Módulo de Configuración Central para el Solucionador de PINNs.
(Con ecuación de calor 2D).

Este módulo centraliza todos los hiperparámetros del modelo, condiciones físicas
y configuraciones de entrenamiento. Actúa como la fuente de la verdad para
el experimento.

Attributes:
    RESULTS_PATH (str): Ruta del directorio donde se guardarán los modelos y gráficos.
    EPOCHS (int): Número predeterminado de iteraciones para el entrenamiento.
    LEARNING_RATE (float): Tasa de aprendizaje para el optimizador Adam.
"""

import numpy as np

# --- Configuración General del Experimento (Compartida) ---
RESULTS_PATH = "results"
EPOCHS = 15000
LEARNING_RATE = 1e-3

# ==============================================================================
# --- DEFINICIÓN DE CONFIGURACIONES PARA CADA PROBLEMA ---
# ==============================================================================

SHO_CONFIG = {
    "RUN_NAME": "Simple_Harmonic_Oscillator",
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,
    
    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1, 
        "output_dim": 1, 
        "num_layers": 5, 
        "hidden_dim": 64, 
        "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi, 
        "t_domain": [0.0, 2.0], 
        "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {
        "n_initial": 1, 
        "n_collocation": 1000
    },
    "LOSS_WEIGHTS": {
        "ode": 1.0, 
        "initial": 100.0
    }
}
"""
dict: Configuración para el Oscilador Armónico Simple (SHO).

Define un sistema masa-resorte sin fricción.
- **PHYSICS_CONFIG**: Define `omega` (frecuencia angular) y dominio temporal.
- **MODEL_CONFIG**: MLP simple con 1 entrada (t) y 1 salida (x).
"""

DHO_CONFIG = {
    "RUN_NAME": "Damped_Harmonic_Oscillator",
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,

    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1, 
        "output_dim": 1, 
        "num_layers": 6, 
        "hidden_dim": 64, 
        "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi, 
        "zeta": 0.1, 
        "t_domain": [0.0, 4.0], 
        "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {
        "n_initial": 1, 
        "n_collocation": 2000
    },
    "LOSS_WEIGHTS": {
        "ode": 1.0, 
        "initial": 100.0
    }
}
"""
dict: Configuración para el Oscilador Armónico Amortiguado (DHO).

Define un sistema masa-resorte con fricción/amortiguamiento.
- **PHYSICS_CONFIG**: Incluye `zeta` (coeficiente de amortiguamiento).
- **MODEL_CONFIG**: Similar al SHO pero ligeramente más profundo (6 capas) para capturar la caída exponencial.
"""

HEAT_CONFIG = {
    "RUN_NAME": "2D_Heat_Equation",
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,

    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 3,           # (x, y, t)
        "output_dim": 1,          # u (temperatura)
        "num_layers": 6, 
        "hidden_dim": 128, 
        "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "alpha": 0.1,             # Coeficiente de difusión térmica
        "x_domain": [0.0, 1.0],   # Dominio espacial en x
        "y_domain": [0.0, 1.0],   # Dominio espacial en y  
        "t_domain": [0.0, 1.0]    # Dominio temporal
    },
    "DATA_CONFIG": {
        "n_initial": 50,         # Puntos en t=0 (condición inicial)
        "n_boundary": 50,        # Puntos en los bordes espaciales
        "n_collocation": 2000    # Puntos internos para la PDE
    },
    "LOSS_WEIGHTS": {
        "pde": 1.0,               # Peso para la pérdida de la PDE
        "initial": 100.0,         # Peso para la condición inicial
        "boundary": 100.0         # Peso para las condiciones de contorno
    }
}
"""
dict: Configuración para la Ecuación de Calor 2D (HEAT).

Modelo PDE dependiente del tiempo en dos dimensiones espaciales.
- **MODEL_CONFIG**: MLP con 3 entradas (x, y, t) y 1 salida (temperatura u).
- **PHYSICS_CONFIG**: Define `alpha` (difusividad térmica) y dominios 2D+T.
- **DATA_CONFIG**: Requiere muestreo de frontera (`n_boundary`) además de inicial y colocación.
"""

# ==============================================================================
# --- EXPORTACIÓN DE LA CONFIGURACIÓN ACTIVA ---
# ==============================================================================

ALL_CONFIGS = {
    "SHO": SHO_CONFIG,
    "DHO": DHO_CONFIG,
    "HEAT": HEAT_CONFIG
}
"""dict: Registro maestro que mapea nombres de problemas a sus diccionarios de configuración."""

def get_active_config(problem_name: str) -> dict:
    """
    Recupera el diccionario de configuración para un problema físico específico.

    Esta función actúa como una fábrica de configuraciones, validando que el
    problema solicitado exista en el registro maestro.

    Args:
        problem_name (str): Identificador del problema. 
                            Opciones válidas: "SHO", "DHO", "HEAT".

    Returns:
        dict: Diccionario de configuración completo conteniendo parámetros del modelo,
              física y entrenamiento.

    Raises:
        ValueError: Si `problem_name` no se encuentra en `ALL_CONFIGS`.
    """
    problem_name = problem_name.upper()
    if problem_name not in ALL_CONFIGS:
        valid_problems = list(ALL_CONFIGS.keys())
        raise ValueError(f"'{problem_name}' no es un problema válido. " 
                        f"Elige entre: {valid_problems}")
    return ALL_CONFIGS[problem_name]