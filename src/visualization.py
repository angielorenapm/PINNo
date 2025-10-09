# src/visualization.py
"""
Visualización de resultados de PINNs entrenadas.

Este script:
- Carga la configuración y el modelo correspondiente.
- Carga los pesos entrenados (best_weights o final_weights).
- Genera una malla de puntos en el dominio temporal o espaciotemporal.
- Muestra y guarda las gráficas de las soluciones predichas.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src import config
from src.models import get_model
from src.physics import get_physics_problem


def visualize_sho(model, physics, t_domain, save_path):
    """Grafica la solución aprendida para el oscilador armónico simple."""
    t = np.linspace(t_domain[0], t_domain[1], 400).reshape(-1, 1).astype(np.float32)
    x_pred = model(t).numpy().flatten()

    plt.figure(figsize=(7, 4))
    plt.plot(t, x_pred, 'b', label="PINN predicha")

    # Solución analítica (si está implementada)
    try:
        x_true = physics.analytical_solution(t).numpy().flatten()
        plt.plot(t, x_true, 'r--', label="Solución analítica")
    except Exception:
        pass

    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Oscilador Armónico Simple (PINN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"[Visualizer] Gráfica guardada en: {save_path}")
    plt.show()


def visualize_wave(model, physics, x_domain, t_domain, save_path):
    """Grafica la solución aprendida para la ecuación de onda 1D."""
    nx, nt = 100, 100
    x = np.linspace(x_domain[0], x_domain[1], nx)
    t = np.linspace(t_domain[0], t_domain[1], nt)
    X, T = np.meshgrid(x, t)
    XT = np.hstack([X.reshape(-1, 1), T.reshape(-1, 1)]).astype(np.float32)

    u_pred = model(XT).numpy().reshape(nt, nx)

    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X, T, u_pred, shading='auto', cmap='viridis')
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Ecuación de Onda 1D - Solución PINN")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"[Visualizer] Gráfica guardada en: {save_path}")
    plt.show()


def main():
    print("--- 🎨 Visualización de PINN ---")

    # Cargar configuración
    cfg = config
    problem = cfg.ACTIVE_PROBLEM
    print(f"[Visualizer] Problema activo: {problem}")

    # Buscar último directorio de resultados
    results_dir = cfg.RESULTS_PATH
    subdirs = sorted(
        [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))],
        key=os.path.getmtime
    )
    if not subdirs:
        raise FileNotFoundError("No se encontraron carpetas de resultados.")
    latest_run = subdirs[-1]
    print(f"[Visualizer] Usando resultados de: {latest_run}")

    # Cargar modelo y construirlo con la forma correcta
    model_config = dict(cfg.MODEL_CONFIG)
    model = get_model(cfg.MODEL_NAME, model_config)

    if problem in ("SHO", "DHO"):
        model.build(input_shape=(None, 1))  # una variable: t
    elif problem == "WAVE":
        model.build(input_shape=(None, 2))  # dos variables: x y t
    else:
        raise ValueError(f"Problema desconocido: {problem}")

    # Cargar pesos
    best_weights = os.path.join(latest_run, "best_weights.weights.h5")
    final_weights = os.path.join(latest_run, "final_weights.weights.h5")

    if os.path.exists(best_weights):
        model.load_weights(best_weights)
        print(f"[Visualizer] Pesos cargados: {best_weights}")
    elif os.path.exists(final_weights):
        model.load_weights(final_weights)
        print(f"[Visualizer] Pesos cargados: {final_weights}")
    else:
        raise FileNotFoundError("No se encontraron archivos de pesos entrenados.")

    model.summary()

    # Cargar física (para posible solución analítica)
    physics = get_physics_problem(problem, {"PHYSICS_CONFIG": cfg.PHYSICS_CONFIG})

    # Definir nombre del archivo de salida
    save_path = os.path.join(latest_run, f"PINN_{problem}_solution.png")

    # Visualizar según el problema
    if problem in ("SHO", "DHO"):
        visualize_sho(model, physics, cfg.PHYSICS_CONFIG["t_domain"], save_path)
    elif problem == "WAVE":
        visualize_wave(model, physics, cfg.PHYSICS_CONFIG["x_domain"], cfg.PHYSICS_CONFIG["t_domain"], save_path)
    else:
        raise ValueError(f"Problema no soportado para visualización: {problem}")

    print("\n--- ✅ Visualización completada ---")


if __name__ == "__main__":
    main()
