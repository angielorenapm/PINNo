# src/training.py

"""
Módulo principal de entrenamiento para PINNs.

Este módulo actúa como el orquestador central (Facade) del sistema. Su responsabilidad
es coordinar la interacción entre los modelos neuronales, la física, los datos
y el cálculo de pérdidas para ejecutar el bucle de entrenamiento.

Patterns:
    - Facade: Proporciona una interfaz unificada para el subsistema de entrenamiento.
    - Strategy: Delega el cálculo de pérdidas y el muestreo de datos.
"""

import os
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, List

# Importaciones relativas para soportar ejecución como paquete
from .models import get_model
from .physics import get_physics_problem
from .data_manage import DataManager
from .losses import LossCalculator


class PINNTrainer:
    """
    Entrenador principal para Redes Neuronales Informadas por la Física.

    Encapsula todo el estado y la lógica necesaria para entrenar un modelo
    desde cero.

    Attributes:
        config (Dict[str, Any]): Configuración completa del experimento.
        active_problem (str): Nombre del problema físico (ej. "SHO").
        epoch (int): Contador actual de épocas de entrenamiento.
        loss_history (List[float]): Registro histórico de la pérdida total.
        model (tf.keras.Model): La red neuronal instanciada.
        physics (PhysicsProblem): El problema físico configurado.
        data_manager (DataManager): El gestor de datos.
        loss_calculator (LossCalculator): La estrategia de cálculo de error.
        optimizer (tf.keras.optimizers.Optimizer): El optimizador (Adam).
        run_dir (str): Ruta donde se guardarán los resultados.
    """
    
    def __init__(self, config_dict: Dict[str, Any], problem_name: str):
        """
        Inicializa el entrenador.

        Args:
            config_dict (Dict[str, Any]): Diccionario de configuración validado.
            problem_name (str): Identificador del problema a resolver.
        """
        # Configuración básica
        self.config = config_dict
        self.active_problem = problem_name
        self.epoch = 0
        self.loss_history = []
        
        # Inicializar componentes especializados
        self._init_components()
        self._setup_experiment()

    def _init_components(self):
        """
        Inicializa los componentes del sistema usando las fábricas correspondientes.
        
        Instancia el Modelo, la Física, el Gestor de Datos y el Calculador de Pérdidas
        basándose en la configuración proporcionada.
        """
        # Modelo
        model_config = dict(self.config["MODEL_CONFIG"])
        self.model = get_model(self.config["MODEL_NAME"], model_config)
        
        # Física
        physics_config = {"PHYSICS_CONFIG": self.config["PHYSICS_CONFIG"]}
        self.physics = get_physics_problem(self.active_problem, physics_config)
        
        # Datos
        self.data_manager = DataManager(self.config, self.active_problem)
        
        # Pérdidas
        self.loss_calculator = LossCalculator(
            self.config["LOSS_WEIGHTS"], 
            self.active_problem
        )
        
        # Optimizador
        self.optimizer = tf.keras.optimizers.Adam(self.config["LEARNING_RATE"])

    def _setup_experiment(self):
        """
        Prepara el entorno del experimento.
        
        Crea el directorio de resultados con marca de tiempo y prepara los
        datos de entrenamiento iniciales.
        """
        # Directorio de resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            self.config["RESULTS_PATH"], 
            f"{self.config['RUN_NAME']}_{timestamp}"
        )
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Preparar datos
        self.data_manager.prepare_data()
        self.training_data = self.data_manager.get_training_data()

    @tf.function
    def train_step(self) -> List[tf.Tensor]:
        """
        Ejecuta un único paso de entrenamiento (Forward + Backward Pass).

        Este método está decorado con `@tf.function` para compilarlo en un grafo
        de TensorFlow, lo que mejora significativamente el rendimiento.

        Returns:
            List[tf.Tensor]: Una lista conteniendo [Pérdida Total, Componente 1, Componente 2...].
        """
        # Obtener datos (tf.function capturará estos tensores como constantes en el grafo
        # a menos que se pasen como argumentos, pero para PINNs estáticas esto suele funcionar)
        training_data = self.training_data
        
        # Forward pass + cálculo de pérdidas
        with tf.GradientTape() as tape:
            total_loss, loss_components = self.loss_calculator.compute_losses(
                self.model, self.physics, training_data
            )
        
        # Backward pass (Cálculo de gradientes y actualización de pesos)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return [total_loss] + loss_components

    def perform_one_step(self) -> List[tf.Tensor]:
        """
        Interfaz pública de alto nivel para ejecutar una iteración.
        
        Llama a `train_step`, actualiza el contador de épocas y registra la
        historia de pérdidas. Es el método que debe llamar la GUI o el CLI loop.

        Returns:
            List[tf.Tensor]: Lista de valores de pérdida del paso actual.
        """
        losses = self.train_step()
        self.epoch += 1
        self.loss_history.append(losses[0].numpy())
        return losses

    def get_training_info(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del estado actual del entrenamiento.

        Returns:
            Dict[str, Any]: Diccionario con época actual, pérdida actual,
                            problema y el historial completo.
        """
        return {
            "epoch": self.epoch,
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "problem": self.active_problem,
            "loss_history": self.loss_history
        }


# ==============================================================================
# --- FUNCIONES AUXILIARES (CLI) ---
# ==============================================================================

def train_complete(config: Dict[str, Any], problem_name: str, epochs: int = None):
    """
    Ejecuta un entrenamiento completo de principio a fin (Modo CLI).

    Esta función bloqueante instancia un entrenador y ejecuta el bucle
    principal hasta completar las épocas especificadas.

    Args:
        config (Dict[str, Any]): Configuración del experimento.
        problem_name (str): Nombre del problema físico.
        epochs (int, optional): Número de épocas a entrenar. Si es None, usa
                                el valor de la configuración.

    Returns:
        PINNTrainer: La instancia del entrenador con el modelo ya entrenado.
    """
    epochs = epochs or config["EPOCHS"]
    
    trainer = PINNTrainer(config, problem_name)
    
    print(f"🚀 Training {problem_name} for {epochs} epochs...")
    print(f"📁 Results will be saved to: {trainer.run_dir}")
    
    for epoch in range(epochs):
        losses = trainer.perform_one_step()
        
        # Logging progresivo en consola
        if epoch == 0 or (epoch + 1) % 500 == 0:
            loss_str = " | ".join([f"Loss_{i}: {loss.numpy():.2e}" 
                                 for i, loss in enumerate(losses)])
            print(f"📊 Epoch {epoch + 1:5d} | {loss_str}")
    
    print("✅ Training completed!")
    return trainer


def main():
    """
    Punto de entrada principal para la ejecución directa del módulo.
    
    Permite probar el entrenamiento rápidamente desde la línea de comandos
    sin invocar la interfaz gráfica.
    """
    try:
        # Intento de importación relativa si se ejecuta como módulo
        from .config import get_active_config
    except ImportError:
        # Fallback a importación absoluta si se ejecuta como script suelto
        from src.config import get_active_config
    
    problem_name = "SHO"  # Por defecto para pruebas rápidas
    config = get_active_config(problem_name)
    
    trainer = train_complete(config, problem_name)
    
    # Información final
    info = trainer.get_training_info()
    print(f"\n📈 Final Epoch: {info['epoch']}")
    print(f"📉 Final Loss: {info['current_loss']:.2e}")


if __name__ == "__main__":
    main()