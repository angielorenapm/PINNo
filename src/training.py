# src/training.py
"""
Módulo principal de entrenamiento para PINNs.

Responsabilidades:
- Coordinar el proceso de entrenamiento
- Gestionar el estado del entrenamiento  
- Proporcionar interfaces para GUI y CLI
"""
import os
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, List

from src.models import get_model
from src.physics import get_physics_problem
from src.data_manage import DataManager
from src.losses import LossCalculator


class PINNTrainer:
    """
    Entrenador principal que orquesta el proceso de entrenamiento.
    
    Pattern: Facade + Coordinator
    """
    
    def __init__(self, config_dict: Dict[str, Any], problem_name: str):
        # Configuración básica
        self.config = config_dict
        self.active_problem = problem_name
        self.epoch = 0
        self.loss_history = []
        
        # Inicializar componentes especializados
        self._init_components()
        self._setup_experiment()

    def _init_components(self):
        """Inicializa los componentes del sistema (Factory Pattern)"""
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
        """Configura el experimento (Template Method)"""
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
        Ejecuta un paso de entrenamiento.
        
        Pattern: Strategy - delega el cálculo de pérdidas a LossCalculator
        """
        # Obtener datos
        training_data = self.training_data
        
        # Forward pass + cálculo de pérdidas
        with tf.GradientTape() as tape:
            total_loss, loss_components = self.loss_calculator.compute_losses(
                self.model, self.physics, training_data
            )
        
        # Backward pass
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return [total_loss] + loss_components

    def perform_one_step(self) -> List[tf.Tensor]:
        """
        Interfaz pública para la GUI.
        
        Pattern: Adapter - mantiene compatibilidad con interfaz existente
        """
        losses = self.train_step()
        self.epoch += 1
        self.loss_history.append(losses[0].numpy())
        return losses

    def get_training_info(self) -> Dict[str, Any]:
        """Información del estado actual del entrenamiento"""
        return {
            "epoch": self.epoch,
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "problem": self.active_problem,
            "loss_history": self.loss_history
        }


# Entrenamiento por lotes para CLI
def train_complete(config: Dict[str, Any], problem_name: str, epochs: int = None):
    """
    Función de alto nivel para entrenamiento completo.
    
    Pattern: Function as Interface
    """
    epochs = epochs or config["EPOCHS"]
    
    trainer = PINNTrainer(config, problem_name)
    
    print(f"🚀 Training {problem_name} for {epochs} epochs...")
    print(f"📁 Results will be saved to: {trainer.run_dir}")
    
    for epoch in range(epochs):
        losses = trainer.perform_one_step()
        
        # Logging progresivo
        if epoch == 0 or (epoch + 1) % 500 == 0:
            loss_str = " | ".join([f"Loss_{i}: {loss.numpy():.2e}" 
                                 for i, loss in enumerate(losses)])
            print(f"📊 Epoch {epoch + 1:5d} | {loss_str}")
    
    print("✅ Training completed!")
    return trainer


# Punto de entrada principal
def main():
    """Función principal para ejecución desde CLI"""
    from src.config import get_active_config
    
    problem_name = "SHO"  # Por defecto, pero podría ser argumento
    config = get_active_config(problem_name)
    
    trainer = train_complete(config, problem_name)
    
    # Información final
    info = trainer.get_training_info()
    print(f"\n📈 Final Epoch: {info['epoch']}")
    print(f"📉 Final Loss: {info['current_loss']:.2e}")


if __name__ == "__main__":
    main()