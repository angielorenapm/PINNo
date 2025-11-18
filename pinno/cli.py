# main.py
"""
Script principal para ejecutar el entrenamiento de una PINN.

Este script importa e invoca la lógica de entrenamiento definida
en el módulo `src.training`. La configuración del experimento
(qué problema resolver, arquitectura del modelo, etc.) se
gestiona centralmente en `src/config.py`.
"""

from .training import main as run_training
import os

def main():
    """Función principal que inicia el experimento."""
    print("--- ✨ Iniciando Experimento PINN ✨ ---")
    
    # Crea el directorio 'results' si no existe
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # Llama a la función principal del módulo de entrenamiento
    run_training()
    
    print("\n--- ✅ Experimento completado exitosamente ---")


if __name__ == "__main__":
    main()