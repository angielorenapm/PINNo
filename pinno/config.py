# pinno/config.py
import numpy as np

# --- Configuracion General ---
RESULTS_PATH = "results"
EPOCHS = 15000
LEARNING_RATE = 1e-3

# Variables para mapeo CSV
PROBLEM_VARIABLES = {
    "SHO": ["time", "displacement"],
    "DHO": ["time", "displacement"], 
    "HEAT": ["x", "y", "time", "temperature"]
}

# ==============================================================================
# CONFIGURACIONES
# ==============================================================================

SHO_CONFIG = {
    "RUN_NAME": "Simple_Harmonic_Oscillator",
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,
    
    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1, "output_dim": 1, 
        "num_layers": 5, "hidden_dim": 64, "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi, 
        "t_domain": [0.0, 2.0],
        # CRITICO: Este diccionario anidado es lo que la GUI lee y modifica
        "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {"n_initial": 1, "n_collocation": 1000},
    "LOSS_WEIGHTS": {"ode": 1.0, "initial": 100.0, "data": 10.0}
}

DHO_CONFIG = {
    "RUN_NAME": "Damped_Harmonic_Oscillator",
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,

    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1, "output_dim": 1, 
        "num_layers": 6, "hidden_dim": 64, "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi, 
        "zeta": 0.1, 
        "t_domain": [0.0, 4.0],
        # CRITICO: Condiciones Iniciales expuestas
        "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {"n_initial": 1, "n_collocation": 2000},
    "LOSS_WEIGHTS": {"ode": 1.0, "initial": 100.0, "data": 10.0}
}

HEAT_CONFIG = {
    "RUN_NAME": "2D_Heat_Equation",
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "RESULTS_PATH": RESULTS_PATH,

    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 3, "output_dim": 1, 
        "num_layers": 6, "hidden_dim": 128, "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "alpha": 0.1, 
        "x_domain": [0.0, 1.0], "y_domain": [0.0, 1.0], "t_domain": [0.0, 1.0]
        # HEAT no suele tener condiciones iniciales escalares simples (son funciones),
        # por lo que no incluimos 'initial_conditions' aqui para evitar confusiones en la GUI.
    },
    "DATA_CONFIG": {"n_initial": 50, "n_boundary": 50, "n_collocation": 500},
    "LOSS_WEIGHTS": {"pde": 1.0, "initial": 100.0, "boundary": 100.0, "data": 10.0}
}

# ==============================================================================
# EXPORTACION
# ==============================================================================

ALL_CONFIGS = {
    "SHO": SHO_CONFIG,
    "DHO": DHO_CONFIG,
    "HEAT": HEAT_CONFIG
}

def get_active_config(problem_name: str) -> dict:
    # Devolvemos una COPIA para que los cambios en la GUI no sean permanentes en la sesion global si reinicias
    import copy
    problem_name = problem_name.upper()
    if problem_name not in ALL_CONFIGS:
        raise ValueError(f"Problema '{problem_name}' no valido.")
    return copy.deepcopy(ALL_CONFIGS[problem_name])

def get_problem_variables(problem_name: str) -> list:
    return PROBLEM_VARIABLES.get(problem_name.upper(), [])