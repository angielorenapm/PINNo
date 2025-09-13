# main.py

"""
Script principal para ejecutar el entrenamiento de una PINN.

Este script realiza los siguientes pasos:
1.  Importa la configuración del experimento desde `src/config.py`.
2.  Crea un directorio único para guardar los resultados de esta ejecución.
3.  Inicializa el modelo, el problema físico y el generador de datos.
4.  Instancia y ejecuta el entrenador.
5.  Guarda el modelo entrenado, el historial de pérdidas y una visualización de la solución.
"""

import os
import pandas as pd
from datetime import datetime

# Asumiendo que has instalado tu proyecto en modo editable (`pip install -e .`)
# o que estás corriendo desde la raíz del proyecto.
from src import config
from src.models import get_model
from src.physics import YourPhysicsProblem  # Reemplaza con tu clase de física
from src.data_generation import DataGenerator # Reemplaza con tu clase de datos
from src.training import Trainer           # Reemplaza con tu clase de entrenamiento
from src.visualization import plot_solution # Reemplaza con tu función de visualización

def run_experiment():
    """Función principal que orquesta todo el experimento."""
    
    # --- 1. Configuración del Entorno ---
    print("--- ⚙️  Configurando el entorno del experimento ---")
    
    # Crear un directorio de resultados único con fecha y hora
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Los resultados se guardarán en: {results_dir}")

    # --- 2. Inicialización de Componentes ---
    print("--- 🧠 Creando el modelo de red neuronal ---")
    
    # Usar la fábrica de modelos de tu `models.py`
    model = get_model(config.MODEL_NAME, config.MODEL_CONFIG)
    model.summary() # Imprime un resumen de la arquitectura del modelo

    print("--- ⚛️  Definiendo el problema físico ---")
    physics_problem = YourPhysicsProblem()

    print("--- 📊 Generando los datos de entrenamiento ---")
    data_generator = DataGenerator()
    training_data = data_generator.generate_points()

    # --- 3. Configuración y Ejecución del Entrenamiento ---
    print("--- 🚀 Iniciando el proceso de entrenamiento ---")
    
    # El entrenador une todos los componentes
    trainer = Trainer(
        model=model,
        physics_problem=physics_problem,
        training_data=training_data,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE
    )
    
    # Iniciar el entrenamiento y obtener el historial de pérdidas
    loss_history = trainer.train()
    
    print("--- ✅ Entrenamiento finalizado ---")

    # --- 4. Guardado de Resultados ---
    print("--- 💾 Guardando los resultados del entrenamiento ---")
    
    # Guardar el modelo en formato SavedModel
    model_path = os.path.join(results_dir, "saved_model")
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Guardar el historial de pérdidas en un archivo CSV
    history_df = pd.DataFrame(loss_history)
    history_path = os.path.join(results_dir, "loss_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Historial de pérdidas guardado en: {history_path}")
    
    # Guardar una visualización de la solución
    plot_path = os.path.join(results_dir, "solution_plot.png")
    plot_solution(model, save_path=plot_path)
    print(f"Gráfica de la solución guardada en: {plot_path}")
    
    print("\n--- ✨ Experimento completado exitosamente ✨ ---")


if __name__ == "__main__":
    run_experiment()