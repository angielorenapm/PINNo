# src/config.py

"""
Módulo de Configuración Central para el Solucionador de PINNs.
(Versión final corregida)
"""
import numpy as np

# --- Configuración General del Experimento (Compartida) ---
RESULTS_PATH = "results"
EPOCHS = 15000
LEARNING_RATE = 1e-3

# ==============================================================================
# --- DEFINICIÓN DE CONFIGURACIONES PARA CADA PROBLEMA ---
# ==============================================================================

# --- 1. Oscilador Armónico Simple (SHO) ---
SHO_CONFIG = {
    "RUN_NAME": "Simple_Harmonic_Oscillator",
    # ¡CORRECCIÓN! Añadir configuraciones generales.
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,
    
    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1, "output_dim": 1, "num_layers": 5, "hidden_dim": 64, "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi, "t_domain": [0.0, 2.0], "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {
        "n_initial": 1, "n_collocation": 1000
    },
    "LOSS_WEIGHTS": {
        "ode": 1.0, "initial": 100.0
    }
}

# --- 2. Oscilador Armónico Amortiguado (DHO) ---
DHO_CONFIG = {
    "RUN_NAME": "Damped_Harmonic_Oscillator",
    # ¡CORRECCIÓN! Añadir configuraciones generales.
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,

    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1, "output_dim": 1, "num_layers": 6, "hidden_dim": 64, "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi, "zeta": 0.1, "t_domain": [0.0, 4.0], "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {
        "n_initial": 1, "n_collocation": 2000
    },
    "LOSS_WEIGHTS": {
        "ode": 1.0, "initial": 100.0
    }
}

# --- 3. Ecuación de Onda 1D (WAVE) ---
WAVE_CONFIG = {
    "RUN_NAME": "1D_Wave_Equation",
    # ¡CORRECCIÓN! Añadir configuraciones generales.
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,

    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 2, "output_dim": 1, "num_layers": 8, "hidden_dim": 128, "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "c": 1.0, "x_domain": [0.0, 1.0], "t_domain": [0.0, 1.0]
    },
    "DATA_CONFIG": {
        "n_initial": 200, "n_boundary": 200, "n_collocation": 20000
    },
    "LOSS_WEIGHTS": {
        "pde": 1.0, "initial": 100.0, "boundary": 100.0
    }
}

# ==============================================================================
# --- EXPORTACIÓN DE LA CONFIGURACIÓN ACTIVA ---
# ==============================================================================

ALL_CONFIGS = {
    "SHO": SHO_CONFIG,
    "DHO": DHO_CONFIG,
    "WAVE": WAVE_CONFIG
}

def get_active_config(problem_name: str) -> dict:
    problem_name = problem_name.upper()
    if problem_name not in ALL_CONFIGS:
        raise ValueError(f"'{problem_name}' no es un problema válido. Elige entre: {list(ALL_CONFIGS.keys())}")
    return ALL_CONFIGS[problem_name]