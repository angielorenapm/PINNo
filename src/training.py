"""
Módulo principal de entrenamiento para PINNs.
(Versión 0.0.5 - Con corrección para Heat Equation y mejoras CSV)
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.models import get_model
from src.physics import get_physics_problem
from src.data_manage import DataManager
from src.losses import LossCalculator


class PINNTrainer:
    """
    Entrenador principal que orquesta el proceso de entrenamiento.
    """
    
    def __init__(self, config_dict: Dict[str, Any], problem_name: str, 
                 csv_data: Optional[pd.DataFrame] = None, 
                 column_mapping: Optional[Dict[str, str]] = None):
        # Configuración básica
        self.config = config_dict
        self.active_problem = problem_name
        self.csv_data = csv_data
        self.column_mapping = column_mapping
        self.use_csv = csv_data is not None and column_mapping is not None
        self.epoch = 0
        self.loss_history = []
        self.learning_rate = self.config["LEARNING_RATE"]
        
        # Inicializar componentes especializados
        self._init_components()
        self._setup_experiment()

    def _init_components(self):
        """Inicializa los componentes del sistema"""
        try:
            # Modelo
            model_config = dict(self.config["MODEL_CONFIG"])
            self.model = get_model(self.config["MODEL_NAME"], model_config)
            
            # Initialize model with a forward pass to build it
            input_dim = model_config["input_dim"]
            
            if self.use_csv:
                # For CSV mode, initialize with data
                if self.active_problem == "HEAT":
                    # For heat equation, we need (x, y, t) data
                    x_data, y_data, t_data, u_data = self._get_csv_training_data_heat()
                    sample_input = tf.concat([x_data[:1], y_data[:1], t_data[:1]], axis=1)
                else:
                    # For SHO/DHO
                    t_data, x_data = self._get_csv_training_data()
                    sample_input = t_data[:1]
                _ = self.model(sample_input)
            else:
                # For analytical mode, initialize with proper domain sample based on input dimension
                if input_dim == 1:
                    # 1D problems (SHO, DHO)
                    t_domain = self.config["PHYSICS_CONFIG"]["t_domain"]
                    sample_input = tf.constant([[t_domain[0]]], dtype=tf.float32)
                elif input_dim == 3:
                    # 3D problems (HEAT) - need (x, y, t)
                    x_domain = self.config["PHYSICS_CONFIG"]["x_domain"]
                    y_domain = self.config["PHYSICS_CONFIG"]["y_domain"]
                    t_domain = self.config["PHYSICS_CONFIG"]["t_domain"]
                    sample_input = tf.constant([[x_domain[0], y_domain[0], t_domain[0]]], dtype=tf.float32)
                else:
                    # Generic fallback
                    sample_input = tf.constant([[0.0] * input_dim], dtype=tf.float32)
                    
                _ = self.model(sample_input)
            
            # Física
            physics_config = {"PHYSICS_CONFIG": self.config["PHYSICS_CONFIG"]}
            self.physics = get_physics_problem(self.active_problem, physics_config, 
                                             self.csv_data, self.column_mapping)
            
            # Datos
            self.data_manager = DataManager(self.config, self.active_problem, 
                                          self.csv_data, self.column_mapping)
            
            # Pérdidas - ajustar pesos para datos CSV
            loss_weights = dict(self.config["LOSS_WEIGHTS"])
            if self.use_csv:
                loss_weights["data"] = 200.0  # Higher weight for CSV data
                # Reduce physics loss weight for CSV data
                if self.active_problem in ["SHO", "DHO"]:
                    loss_weights["ode"] = 0.1
                else:
                    loss_weights["pde"] = 0.1
            
            self.loss_calculator = LossCalculator(loss_weights, self.active_problem)
            
            # Optimizador con learning rate decay
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=self.learning_rate,
                    decay_steps=1000,
                    decay_rate=0.95
                )
            )
            
            print(f"Components initialized for {self.active_problem} - Mode: {'CSV' if self.use_csv else 'Analytical'}")
            print(f"Model input dimension: {input_dim}")
            
        except Exception as e:
            print(f"Error initializing components: {e}")
            raise

    def _get_csv_training_data(self):
        """Get CSV data for SHO/DHO model initialization"""
        if not self.use_csv or self.csv_data is None:
            return None, None
            
        time_col = self.column_mapping['time']
        disp_col = self.column_mapping['displacement']
        
        t_data = self.csv_data[time_col].values.reshape(-1, 1)
        x_data = self.csv_data[disp_col].values.reshape(-1, 1)
        
        return tf.constant(t_data, dtype=tf.float32), tf.constant(x_data, dtype=tf.float32)

    def _get_csv_training_data_heat(self):
        """Get CSV data for heat equation initialization"""
        if not self.use_csv or self.csv_data is None:
            return None, None, None, None
            
        x_col = self.column_mapping['x']
        y_col = self.column_mapping['y'] 
        time_col = self.column_mapping['time']
        temp_col = self.column_mapping['temperature']
        
        x_data = self.csv_data[x_col].values.reshape(-1, 1)
        y_data = self.csv_data[y_col].values.reshape(-1, 1)
        t_data = self.csv_data[time_col].values.reshape(-1, 1)
        u_data = self.csv_data[temp_col].values.reshape(-1, 1)
        
        return (tf.constant(x_data, dtype=tf.float32), 
                tf.constant(y_data, dtype=tf.float32),
                tf.constant(t_data, dtype=tf.float32),
                tf.constant(u_data, dtype=tf.float32))

    def _setup_experiment(self):
        """Configura el experimento"""
        try:
            # Directorio de resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode = "csv" if self.use_csv else "analytical"
            self.run_dir = os.path.join(
                self.config["RESULTS_PATH"], 
                f"{self.config['RUN_NAME']}_{mode}_{timestamp}"
            )
            os.makedirs(self.run_dir, exist_ok=True)
            
            # Preparar datos
            self.data_manager.prepare_data()
            self.training_data = self.data_manager.get_training_data()
            
            print(f"Experiment setup complete - Results dir: {self.run_dir}")
            
        except Exception as e:
            print(f"Error setting up experiment: {e}")
            raise

    def train_step(self) -> List[tf.Tensor]:
        """
        Ejecuta un paso de entrenamiento.
        """
        try:
            training_data = self.training_data
            
            with tf.GradientTape() as tape:
                total_loss, loss_components = self.loss_calculator.compute_losses(
                    self.model, self.physics, training_data
                )
            
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            
            # Apply gradient clipping to prevent exploding gradients
            grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
            
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            return [total_loss] + loss_components
            
        except Exception as e:
            print(f"Error in train step: {e}")
            # Return high loss to indicate problem but prevent crash
            return [tf.constant(1e6, dtype=tf.float32)] * 3

    def perform_one_step(self) -> List[tf.Tensor]:
        """
        Interfaz pública para la GUI.
        """
        losses = self.train_step()
        self.epoch += 1
        
        # Only append if we have a valid loss
        if len(losses) > 0 and losses[0] is not None:
            try:
                loss_value = losses[0].numpy()
                if not np.isnan(loss_value) and not np.isinf(loss_value):
                    self.loss_history.append(loss_value)
            except:
                pass  # Skip if we can't convert to numpy
                
        return losses

    def get_training_info(self) -> Dict[str, Any]:
        """Información del estado actual del entrenamiento"""
        return {
            "epoch": self.epoch,
            "current_loss": self.loss_history[-1] if self.loss_history else None,
            "problem": self.active_problem,
            "mode": "csv" if self.use_csv else "analytical",
            "loss_history": self.loss_history,
            "column_mapping": self.column_mapping
        }

    def switch_to_analytical_mode(self):
        """Switch from CSV mode to analytical mode"""
        if self.use_csv:
            self.use_csv = False
            self.csv_data = None
            self.column_mapping = None
            self.physics.has_analytical = True
            
            # Reinitialize components
            self._init_components()
            self._setup_experiment()


# Entrenamiento por lotes para CLI
def train_complete(config: Dict[str, Any], problem_name: str, 
                  csv_data: Optional[pd.DataFrame] = None, 
                  column_mapping: Optional[Dict[str, str]] = None,
                  epochs: int = None):
    """
    Función de alto nivel para entrenamiento completo.
    """
    epochs = epochs or config["EPOCHS"]
    
    trainer = PINNTrainer(config, problem_name, csv_data, column_mapping)
    
    mode = "CSV data" if csv_data is not None else "analytical solution"
    print(f"Training {problem_name} for {epochs} epochs using {mode}...")
    if column_mapping:
        print(f"Column mapping: {column_mapping}")
    print(f"Results will be saved to: {trainer.run_dir}")
    
    for epoch in range(epochs):
        losses = trainer.perform_one_step()
        
        if epoch == 0 or (epoch + 1) % 500 == 0:
            loss_str = " | ".join([f"Loss_{i}: {loss.numpy():.2e}" 
                                 for i, loss in enumerate(losses)])
            print(f"Epoch {epoch + 1:5d} | {loss_str}")
    
    print("Training completed!")
    return trainer


# Punto de entrada principal
def main():
    """Función principal para ejecución desde CLI"""
    from src.config import get_active_config
    
    problem_name = "SHO"  # Por defecto
    config = get_active_config(problem_name)
    
    # Para CLI, no hay datos CSV por defecto
    trainer = train_complete(config, problem_name)
    
    info = trainer.get_training_info()
    print(f"\n Final Epoch: {info['epoch']}")
    print(f"Final Loss: {info['current_loss']:.2e}")
    print(f"Mode: {info['mode']}")


if __name__ == "__main__":
    main()