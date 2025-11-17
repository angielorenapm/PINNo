#src/config.py
"""
Módulo de Configuración Central para el Solucionador de PINNs.
(Versión 0.0.4 - Con soporte para mapeo de columnas CSV)
"""
import numpy as np

# --- Configuración General del Experimento (Compartida) ---
RESULTS_PATH = "results"
EPOCHS = 15000
LEARNING_RATE = 1e-3

# --- Variables requeridas para cada problema ---
PROBLEM_VARIABLES = {
    "SHO": ["time", "displacement"],
    "DHO": ["time", "displacement"], 
    "HEAT": ["x", "y", "time", "temperature"]
}

# ==============================================================================
# --- DEFINICIÓN DE CONFIGURACIONES PARA CADA PROBLEMA ---
# ==============================================================================

# --- 1. Oscilador Armónico Simple (SHO) ---
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
        "initial": 100.0,
        "data": 10.0
    }
}

# --- 2. Oscilador Armónico Amortiguado (DHO) ---
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
        "initial": 100.0,
        "data": 10.0
    }
}

# --- 3. Ecuación de Calor 2D (HEAT) ---
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
        "n_collocation": 500    # Puntos internos para la PDE
    },
    "LOSS_WEIGHTS": {
        "pde": 1.0,               # Peso para la pérdida de la PDE
        "initial": 100.0,         # Peso para la condición inicial
        "boundary": 100.0,        # Peso para las condiciones de contorno
        "data": 10.0              # Peso para datos CSV
    }
}

# ==============================================================================
# --- EXPORTACIÓN DE LA CONFIGURACIÓN ACTIVA ---
# ==============================================================================

ALL_CONFIGS = {
    "SHO": SHO_CONFIG,
    "DHO": DHO_CONFIG,
    "HEAT": HEAT_CONFIG
}

def get_active_config(problem_name: str) -> dict:
    """
    Obtiene la configuración activa para un problema dado.
    """
    problem_name = problem_name.upper()
    if problem_name not in ALL_CONFIGS:
        valid_problems = list(ALL_CONFIGS.keys())
        raise ValueError(f"'{problem_name}' no es un problema válido. " 
                        f"Elige entre: {valid_problems}")
    return ALL_CONFIGS[problem_name]

def get_problem_variables(problem_name: str) -> list:
    """
    Obtiene las variables requeridas para un problema específico.
    """
    problem_name = problem_name.upper()
    return PROBLEM_VARIABLES.get(problem_name, [])