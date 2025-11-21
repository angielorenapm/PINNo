# PINNo — Physics-Informed Neural Network (PINN) Interactive Trainer

**Status:** v1.0 - Initial Release
**Authors:** Angie Lorena Pineda Morales, Juan Sebastian Acuña Tellez, Pablo Patiño Bonilla.

This document explains how to install, run and use the PINNo application, the sample models and problems included, and serves as a step-by-step tutorial for new users.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Project layout (quick)](#project-layout-quick)
5. [Running the GUI (Tutorial)](#running-the-gui-tutorial)
6. [Running training from the command line / scripts](#running-training-from-the-command-line--scripts)
7. [Configuration & problems (formats and examples)](#configuration--problems-formats-and-examples)
8. [Sample models provided](#sample-models-provided)
9. [What the GUI displays / how to interpret results](#what-the-gui-displays--how-to-interpret-results)
10. [Running tests](#running-tests)
11. [Extending PINNo (adding models or physics)](#extending-pinno-adding-models-or-physics)
12. [Troubleshooting & common errors](#troubleshooting--common-errors)
13. [License & contact](#license--contact)

---

## Project overview

PINNo is a modular PINN framework designed for rapid experimentation with physics problems expressed as ODEs/PDEs. This version (v1.0) introduces a complete interactive environment for learning, training, and exporting scientific models.

Key features:
* **Interactive GUI**: 4 tabs covering theory, data loading, training, and reporting.
* **Dynamic Configuration**: Modify neural network architecture (layers, neurons, activation) without touching code.
* **Advanced Visualization**: Real-time 1D slices and 2D Heatmaps for the Heat Equation.
* **Export Capabilities**: Save your trained models (.keras), data (.csv), and plots (.png).

---

## Requirements

Tested with Python 3.10+. Required packages (see `requirements.txt`):

tensorflow>=2.12.0
numpy>=1.24
matplotlib>=3.7
pandas>=1.5.0
openpyxl>=3.1.0
scipy>=1.10.0
pytest>=7.0

Use a virtual environment (venv or conda) and install via pip.

---

## Installation

1. Create and activate a virtual environment:

# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate

2. Install dependencies:

pip install -r requirements.txt

---

## Project layout 

Important files and modules you will use:

.
├─ main.py                # Top-level CLI runner
├─ gui.py                 # Main GUI entry point
├─ requirements.txt
├─ src/
│  ├─ config.py           # Central configuration (SHO, DHO, HEAT)
│  ├─ models.py           # Model factory (PINN_MLP)
│  ├─ physics.py          # Physics definitions & Residuals
│  ├─ training.py         # PINNTrainer logic
│  ├─ data_manage.py      # Data sampling & CSV loading
│  ├─ losses.py           # Loss calculation strategies
│  └─ gui_modules/        # Modular GUI components
│     ├─ info_tab.py      # Educational content
│     ├─ training_tab.py  # Training loop & Visualization
│     └─ report_tab.py    # Metrics & Exporting
└─ tests/                 # Pytest suite

---

## Running the GUI (Tutorial)

This section serves as a quick tutorial to get started with PINNo using the graphical interface.

1. **Launch the Application**:
   Run the following command in your terminal:
   
   python gui.py

2. **Step 1: Learn the Basics (Tab: Learn PINNs)**
   * Navigate to the first tab to understand the difference between standard Neural Networks and PINNs.
   * Review the "Hyperparameters Guide" to understand what Learning Rate and Epochs do.

3. **Step 2: Load Experimental Data (Tab: Data Exploration) [Optional]**
   * If you have a `.csv` file (e.g., columns `t`, `x`), click "Load Data".
   * The app will plot your data. This activates "Hybrid Training", where the network learns from both your data AND the physics equation.

4. **Step 3: Configure and Train (Tab: Train & Config)**
   * **Select Problem**: Choose `SHO` (Oscillator), `DHO` (Damped), or `HEAT` (2D Heat Eq).
   * **Customize Architecture**: 
     * Try changing `Activation` to "tanh" (best for physics).
     * For `HEAT`, increase `Hidden Layers` to 6.
   * **Modify Physics**: You can change parameters like Frequency (Omega) directly in the inputs.
   * **Action**: Click **Start Training**.
   * **Observe**: Watch the "Loss" graph decrease. For `HEAT`, watch the bottom heatmaps evolve in real-time to match the analytical solution.

5. **Step 4: Analyze and Export (Tab: Metrics Report)**
   * Click **🔄 Generar Reporte** to fetch the latest statistics.
   * **Save Results**:
     * Click "💾 Guardar Gráfica" to save the convergence plot.
     * Click "🧠 Guardar Modelo" to save the `.keras` file for future use.

---

## Running training from the command line / scripts

For long experiments on servers without a screen, use the CLI.

### A — Use `main.py` (CLI)

This script runs the training using the default settings defined in `src/config.py`.

python main.py

It will create a timestamped folder in `results/` containing the model checkpoints.

### B — Run programmatically

You can customize the training script in Python:

from src.config import get_active_config
from src.training import PINNTrainer

# Load default config
cfg = get_active_config("SHO")

# Modify config programmatically
cfg["EPOCHS"] = 5000
cfg["LEARNING_RATE"] = 0.001

# Run
trainer = PINNTrainer(cfg, "SHO")
for epoch in range(5000):
    trainer.perform_one_step()

---

## Configuration & problems (formats and examples)

All problem configs are centralized in `src/config.py`. Use the helper:

from src.config import get_active_config
cfg = get_active_config("SHO")

Important sections inside the returned config dict:

* `MODEL_CONFIG`: dict with `num_layers`, `hidden_dim`, `activation`.
* `PHYSICS_CONFIG`: physical parameters (e.g., `omega`, `zeta`, `alpha`).
* `DATA_CONFIG`: numbers of collocation points, initial points, and boundary points.
* `LOSS_WEIGHTS`: weightings for the components of the loss (ODE/PDE, Initial, Data).

**Note:** The GUI overrides these values at runtime based on the input fields in the "Train & Config" tab.

---

## Sample models provided

* **PINN_MLP** (in `src/models.py`): a dense feed-forward network.

**Model input shapes:**

* For SHO/DHO: `input_dim = 1` (time `t`).
* For HEAT: `input_dim = 3` (space and time `x, y, t`).

All tensors should be `float32`.

---

## What the GUI displays / how to interpret results

1. **Loss plot** (Top):
   * Shows the Log-Scale Loss vs. Epochs.
   * A downward slope indicates the network is learning. Spikes usually mean the "Learning Rate" is too high.

2. **Solution plot** (Bottom):
   * **SHO/DHO**: Shows position `x` over time `t`.
     * **Dotted Line**: Analytical (Real) solution.
     * **Solid Line**: Neural Network prediction.
   * **HEAT**: 
     * **Middle**: 1D slice of temperature at the center of the domain.
     * **Bottom**: Real-time 2D Heatmaps (Left: Predicted, Right: Analytical).

---

## Running tests

A robust test suite using `pytest` verifies physics residuals, data loading, and architecture gradients.

From the project root:

pytest

To see code coverage:

pytest --cov=src tests/

---

## Extending PINNo (adding models or physics)

**Add a new model**
1. Implement a new `tf.keras.Model` in `src/models.py`.
2. Register it in the `MODELS` dict.

**Add a new physics problem**
1. Create a new class inheriting from `PhysicsProblem` in `src/physics.py`.
2. Implement `pde_residual` (using `GradientTape`) and `analytical_solution`.
3. Add the new class to the `PROBLEMS` mapping.
4. Add a matching config to `src/config.py`.

---

## Troubleshooting & common errors

### `x_tt is None` or derivative errors
Symptoms: `ValueError: The calculation of the derivative failed`.
Fix: Ensure you are using a differentiable activation function like `tanh` or `swish`. `relu` has zero second derivatives in many regions.

### High Loss in HEAT equation
Symptoms: The model does not converge or shows a flat heatmap.
Fix: The 2D Heat equation is complex. Increase epochs to 20,000+ and network depth to 6+ layers in the GUI configuration.

### GUI Freezes
Fix: The training runs in a separate thread, but plotting heavy heatmaps every epoch can slow things down. The GUI updates graphs every 10 epochs to prevent this.

---

## License & contact

This repository contains research code developed at Universidad Distrital Francisco José de Caldas.

**Authors:**
* Angie Lorena Pineda Morales (alpinedam@udistrital.edu.co)
* Pablo Patiño Bonilla (jppatinob@udistrital.edu.co)
* Juan Sebastian Acuña Tellez (jsacunat@udistrital.edu.co

Good luck experimenting with PINNo v1.0.
