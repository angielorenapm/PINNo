# src/config.py

"""
Módulo de Configuración Central para el Solucionador de PINNs.

Este archivo contiene las configuraciones para múltiples problemas físicos.
Para cambiar de problema, simplemente modifica la variable `ACTIVE_PROBLEM`.
"""
import numpy as np

# --- Selector de Problema Activo ---
# Cambia este valor a "SHO", "DHO", o "WAVE" para seleccionar el problema.
ACTIVE_PROBLEM = "SHO"

# --- Configuración General del Experimento (Compartida) ---
RESULTS_PATH = "results"
EPOCHS = 15000
LEARNING_RATE = 1e-3

# ==============================================================================
# --- DEFINICIÓN DE CONFIGURACIONES PARA CADA PROBLEMA ---
# ==============================================================================

# --- 1. Oscilador Armónico Simple (SHO) ---
# Ecuación: d²x/dt² + ω²x = 0
SHO_CONFIG = {
    "RUN_NAME": "Simple_Harmonic_Oscillator",
    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1,      # Entrada: t
        "output_dim": 1,     # Salida: x(t)
        "num_layers": 5,
        "hidden_dim": 64,
        "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi,  # Frecuencia angular (ω)
        "t_domain": [0.0, 2.0],  # Dominio temporal (2 periodos)
        # Condiciones iniciales: x(0)=1, x'(0)=0
        "initial_conditions": {"x0": 1.0, "v0": 0.0}
    },
    "DATA_CONFIG": {
        "n_initial": 1,        # Solo 1 punto en t=0 para la condición inicial
        "n_collocation": 1000  # Puntos para evaluar la Ecuación Diferencial
    },
    "LOSS_WEIGHTS": {
        "ode": 1.0,        # Peso para la Ecuación Diferencial Ordinaria (EDO)
        "initial": 100.0   # Peso para las condiciones iniciales
    }
}

# --- 2. Oscilador Armónico Amortiguado (DHO) ---
# Ecuación: d²x/dt² + 2ζω(dx/dt) + ω²x = 0
DHO_CONFIG = {
    "RUN_NAME": "Damped_Harmonic_Oscillator",
    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 1,      # Entrada: t
        "output_dim": 1,     # Salida: x(t)
        "num_layers": 6,
        "hidden_dim": 64,
        "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "omega": 2 * np.pi,    # Frecuencia angular (ω)
        "zeta": 0.1,           # Coeficiente de amortiguamiento (ζ)
        "t_domain": [0.0, 4.0],  # Dominio temporal (más largo para ver el decaimiento)
        # Condiciones iniciales: x(0)=1, x'(0)=0
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

# --- 3. Ecuación de Onda 1D (WAVE) ---
# Ecuación: ∂²u/∂t² = c² * ∂²u/∂x²
WAVE_CONFIG = {
    "RUN_NAME": "1D_Wave_Equation",
    "MODEL_NAME": "mlp",
    "MODEL_CONFIG": {
        "input_dim": 2,      # Entradas: x, t
        "output_dim": 1,     # Salida: u(x,t)
        "num_layers": 8,
        "hidden_dim": 128,
        "activation": "tanh"
    },
    "PHYSICS_CONFIG": {
        "c": 1.0,            # Velocidad de la onda
        "x_domain": [0.0, 1.0],  # Dominio espacial
        "t_domain": [0.0, 1.0]   # Dominio temporal
    },
    "DATA_CONFIG": {
        "n_initial": 200,      # Puntos para u(x,0) y u_t(x,0)
        "n_boundary": 200,     # Puntos en x=0 y x=1
        "n_collocation": 20000 # Puntos para evaluar la Ecuación Diferencial Parcial (EDP)
    },
    "LOSS_WEIGHTS": {
        "pde": 1.0,        # Peso para la EDP
        "initial": 100.0,  # Peso para la condición inicial
        "boundary": 100.0  # Peso para las condiciones de borde
    }
}

# ==============================================================================
# --- EXPORTACIÓN DE LA CONFIGURACIÓN ACTIVA ---
# ==============================================================================

# Diccionario maestro de configuraciones
ALL_CONFIGS = {
    "SHO": SHO_CONFIG,
    "DHO": DHO_CONFIG,
    "WAVE": WAVE_CONFIG
}

# Seleccionar la configuración activa
if ACTIVE_PROBLEM not in ALL_CONFIGS:
    raise ValueError(f"'{ACTIVE_PROBLEM}' no es un problema válido. Elige entre: {list(ALL_CONFIGS.keys())}")

_active_config = ALL_CONFIGS[ACTIVE_PROBLEM]

# Exportar las variables que usarán los otros módulos
RUN_NAME = _active_config["RUN_NAME"]
MODEL_NAME = _active_config["MODEL_NAME"]
MODEL_CONFIG = _active_config["MODEL_CONFIG"]
PHYSICS_CONFIG = _active_config["PHYSICS_CONFIG"]
DATA_CONFIG = _active_config["DATA_CONFIG"]
LOSS_WEIGHTS = _active_config["LOSS_WEIGHTS"]