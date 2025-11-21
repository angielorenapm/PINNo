# pinno/training.py
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from .models import get_model
from .physics import get_physics_problem
from .data_manage import DataManager
from .losses import LossCalculator

class PINNTrainer:
    def __init__(self, config_dict: Dict[str, Any], problem_name: str, 
                 csv_data: Optional[pd.DataFrame] = None, 
                 column_mapping: Optional[Dict[str, str]] = None):
        self.config = config_dict
        self.active_problem = problem_name
        self.csv_data = csv_data
        self.column_mapping = column_mapping
        
        # Determinar modo
        self.use_csv = (csv_data is not None) and (column_mapping is not None)
        
        self.epoch = 0
        self.loss_history = []
        
        self._init_components()
        self._setup_experiment()

    def _init_components(self):
        # 1. Modelo
        model_config = dict(self.config["MODEL_CONFIG"])
        self.model = get_model(self.config["MODEL_NAME"], model_config)
        self.model(tf.zeros((1, model_config["input_dim"]))) # Init weights
        
        # 2. Fisica
        physics_config = {"PHYSICS_CONFIG": self.config["PHYSICS_CONFIG"]}
        self.physics = get_physics_problem(self.active_problem, physics_config, 
                                         self.csv_data, self.column_mapping)
        
        # --- FIX PARA ERROR KEYERROR t0 ---
        if self.use_csv:
            self.physics.has_analytical = False
            print(f"DEBUG: CSV Mode enabled for {self.active_problem}")
        
        # 3. Datos
        self.data_manager = DataManager(self.config, self.active_problem, 
                                      self.csv_data, self.column_mapping)
        
        # 4. Perdidas
        loss_weights = dict(self.config["LOSS_WEIGHTS"])
        if self.use_csv:
            loss_weights["data"] = 200.0
            key = "ode" if self.active_problem in ["SHO", "DHO"] else "pde"
            loss_weights[key] = 0.1
            
        self.loss_calculator = LossCalculator(loss_weights, self.active_problem)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["LEARNING_RATE"])

    def _setup_experiment(self):
        mode = "csv" if self.use_csv else "analytical"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config["RESULTS_PATH"], f"{self.config['RUN_NAME']}_{mode}_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.data_manager.prepare_data()
        self.training_data = self.data_manager.get_training_data()

    @tf.function
    def train_step(self) -> List[tf.Tensor]:
        with tf.GradientTape() as tape:
            total, components = self.loss_calculator.compute_losses(self.model, self.physics, self.training_data)
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return [total] + components

    def perform_one_step(self) -> List[tf.Tensor]:
        losses = self.train_step()
        self.epoch += 1
        self.loss_history.append(losses[0].numpy())
        return losses

    def get_training_info(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "loss": self.loss_history[-1] if self.loss_history else 0,
            "problem": self.active_problem
        }