# PINNo — Physics-Informed Neural Network (PINN) App

**Tutorial & User Guide (GUI-focused)**

**Status:** formal user-facing README.
This document explains how to install, run and use the PINNo application (graphical and programmatic entry points), the sample models and problems included, the expected data / config formats, where results are saved, common troubleshooting steps, and how to extend the codebase.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Project layout (quick)](#project-layout-quick)
5. [Running the GUI (recommended)](#running-the-gui-recommended)
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

PINNo is a small, modular PINN framework designed for rapid experimentation with physics problems expressed as ODEs/ PDEs. The repository provides:

* a **GUI** (`gui.py`) for interactive training and live plotting,
* a modular `src/` package with `config`, `models`, `physics`, `training`, `data_generation` helpers and `visualization`,
* three sample physics problems: **SHO** (Simple Harmonic Oscillator), **DHO** (Damped Harmonic Oscillator) and **WAVE** (1D wave equation),
* a sample MLP-based PINN architecture (`PINN_MLP`) implemented in TensorFlow/Keras.

The GUI is the recommended entry point for new users: select a problem, start/stop training, and watch the loss and analytic vs predicted solution plots live.

---

## Requirements

Tested with Python 3.10+. Required packages (see `requirements.txt` excerpt):

```
tensorflow>=2.12.0
numpy>=1.24
matplotlib>=3.7
pyyaml>=6.0
pytest>=7.0
tensorboard>=2.12
black
ruff
typing-extensions
tqdm
pandas
```

Use a virtual environment (venv or conda) and install via pip.

---

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows (PowerShell)
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

(If you do not have a `requirements.txt`, install the packages listed above manually.)

---

## Project layout 

Important files and modules you will use:

```
.
├─ main.py                # top-level runner (calls training main if provided)
├─ gui.py                 # Tkinter GUI recommended for interactive use
├─ requirements.txt
├─ src/
│  ├─ config.py           # central configuration (get_active_config)
│  ├─ models.py           # model factory (PINN_MLP)
│  ├─ physics.py          # PhysicsProblem classes (SHO, DHO, WAVE)
│  ├─ training.py         # PINNTrainer (performs training step(s))
│  ├─ data_generation.py  # sampling helpers
│  └─ visualization.py    # plotting helpers
└─ tests/
   └─ structure_nn_test.py
```

---

## Running the GUI (recommended)

The GUI is implemented in `gui.py` and provides an interactive training loop with live plots.

1. Launch the GUI:

```bash
python gui.py
```

2. GUI controls:

* **Select problem**: Choose one of `SHO`, `DHO`, or `WAVE` from the problem dropdown.
* **Start training**: Initializes a `PINNTrainer` with the chosen problem configuration and begins iterative training. The GUI calls `perform_one_step()` repeatedly; training is driven by the GUI event loop.
* **Stop training**: Safely stops the GUI-driven loop.
* **Metrics panel**: shows current epoch, current loss (log scale), and relative L2 error (analytical vs predicted) updated periodically.
* **Plots panel**:

  * Top subplot: loss history (log scale) vs epochs.
  * Bottom subplot: predicted solution vs analytical solution for the chosen problem. For SHO/DHO the plot shows `x(t)` across the configured time domain; for WAVE it shows a spatial slice at `t = 0.5`.

3. Notes:

* The GUI initializes randomness seeds (`np.random.seed(42)` and `tf.random.set_seed(42)` by default in `gui.py`) for reproducibility.
* If initialization fails, an error dialog is displayed with the exception message.

---

## Running training from the command line / scripts

Two options:

### A — Use `main.py`

`main.py` intends to call a top-level `run_training()` from `src.training`. If `src.training.main` exists:

```bash
python3 gui.py
```

This will run the packaged training routine and create a `results/` folder (if not present).

### B — Run programmatically (recommended if no `training.main`)

You can run training headlessly by creating a `PINNTrainer` and advancing it in a loop. Example:

```python
from src.config import get_active_config
from src.training import PINNTrainer

cfg = get_active_config("SHO")        # or "DHO" / "WAVE"
trainer = PINNTrainer(cfg, "SHO")    # construct trainer

n_epochs = int(cfg.get("EPOCHS", 1000))
for epoch in range(n_epochs):
    outputs = trainer.perform_one_step()   # returns loss tensors
    total_loss = outputs[0].numpy()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={total_loss:.4e}")
```

**Where results are stored:** `PINNTrainer` creates a run directory under `cfg["RESULTS_PATH"]` with a timestamp (e.g. `results/Simple_Harmonic_Oscillator_20251015_121010`). Use that directory to save weights or artifacts.

---

## Configuration & problems (formats and examples)

All problem configs are centralized in `src/config.py`. Use the helper:

```python
from src.config import get_active_config
cfg = get_active_config("SHO")   # returns a dict with keys: RUN_NAME, MODEL_NAME, MODEL_CONFIG, PHYSICS_CONFIG, DATA_CONFIG, LOSS_WEIGHTS, EPOCHS, LEARNING_RATE, RESULTS_PATH
```

Important sections inside the returned config dict:

* `MODEL_NAME`: string key used by `src.models.get_model` (currently `"mlp"`).
* `MODEL_CONFIG`: dict with `input_dim`, `output_dim`, `num_layers`, `hidden_dim`, `activation`.
* `PHYSICS_CONFIG`: physical parameters for the chosen problem (e.g., `omega`, `zeta`, `t_domain`, `x_domain`, `c`) and initial conditions.
* `DATA_CONFIG`: numbers of collocation points, initial points and boundary points.
* `LOSS_WEIGHTS`: weightings for the components of the loss (e.g. `ode` / `pde`, `initial`, `boundary`).

To change parameters for an experiment, either:

* edit `src/config.py` directly, or
* load the dict programmatically and modify before constructing `PINNTrainer`.

---

## Sample models provided

* **PINN_MLP** (in `src/models.py`): a dense feed-forward network implemented as a `tf.keras.Model`. `get_model("mlp", model_config)` returns a `PINN_MLP` instance configured by `model_config`.

**Model input shapes:**

* For SHO/DHO: `input_dim = 1` and model expects input shape `(N, 1)` representing time `t`.
* For WAVE: `input_dim = 2` and model expects input shape `(N, 2)` representing `[x, t]`.

All tensors should be `float32`.

---

## What the GUI displays — interpreting outputs

1. **Loss plot** (top): log-scale loss vs epochs. The loss is typically a weighted sum of:

   * PDE residual loss (MSE of residuals),
   * initial condition losses,
   * boundary condition losses (for PDEs).
2. **Solution plot** (bottom):

   * SHO/DHO: predicted `x(t)` vs analytic solution `x_true(t)`. A small relative L2 error indicates a good fit.
   * WAVE: predicted `u(x, t0)` vs analytic solution at a fixed `t0` slice.
3. **Metrics**:

   * Epoch: number of steps performed by the GUI loop.
   * Loss: the most recent scalar total loss.
   * Relative L2 error: `||pred - true|| / ||true||` (small is better).

---

## Running tests

A small test file `tests/structure_nn_test.py` verifies model creation and forward pass shapes.

From the project root:

```bash
pytest -q
```

Make sure the repo root is in `PYTHONPATH` (the test file adds it automatically in the provided test helper).

---

## Extending PINNo — adding models or physics

**Add a new model**

1. Implement a new `tf.keras.Model` subclass in `src/models.py` (or a new module under `src/`).
2. Register it in the `MODELS` dict or extend `get_model()` accordingly.
3. Add a new `MODEL_CONFIG` to `src/config.py` for experiments that use the model.

**Add a new physics problem**

1. Create a new class inheriting from `PhysicsProblem` in `src/physics.py`.
2. Implement:

   * `pde_residual(self, model, points)` — returns a Tensor of residuals (shape `(N, 1)`) using TensorFlow `GradientTape` correctly (watch inputs, compute first and second derivatives as needed).
   * `analytical_solution(self, points)` — optional but useful for GUI comparison.
3. Add the new class to the `PROBLEMS` mapping with a key (e.g., `"MYPROB"`).
4. Add a matching problem config to `src/config.py` and use `get_active_config("MYPROB")`.

**Design notes:** the codebase is set up to evolve toward a Strategy/Registry pattern (so you can register new models/problems without editing central decision logic).

---

## Troubleshooting & common errors

### `None` returned for derivatives / `x_tt` is None

Symptoms: exception raised like `ValueError: The calculation of the second derivative (x_tt) failed and returned None`.

Possible causes & fixes:

* **Input shaped incorrectly**: ensure inputs are `tf.Tensor` with shape `(N, 1)` for time-only problems or `(N, 2)` for `[x,t]`.
* **Not watching the correct tensor**: the physics implementation must call `tape.watch(t)` (or `xt`) before calling `model(t)`.
* **Model not differentiable with respect to inputs**: built-in Keras layers are differentiable but ensure inputs are floating point `tf.float32` and model is applied to the watched tensors directly.
* **Using numpy arrays inside the gradient context**: convert to `tf.Tensor` before `GradientTape`.
* **Eager vs @tf.function** mismatch**: sometimes wrapping methods in `@tf.function` affects shapes/trace; use print/debugging in eager mode first.

### GPU / memory issues

* TensorFlow memory growth can be set at start to avoid OOM. This project currently uses CPU by default. Configure a GPU runtime if desired.

### `main.py` failing to find `training.main`

* Some repository versions expose `PINNTrainer` class without a `main()` free function. If `main.py` fails, use `gui.py` or see the programmatic example above to run training directly.

---

## Notes for advanced users / developers

* The architecture favors modularity (models, physics, data generation, trainer). Consider refactoring to a registry and callback-based Trainer for greater extensibility (Strategy, Factory, Callback patterns).
* Add checkpointing (model + optimizer + config manifest) for long experiments if needed.

---

## License & contact

This repository contains example research code. Check the top-level `LICENSE` file for licensing details.

Authory:
Angie Lorena Pineda Morales alpinedam@udistrital.edu.co
Juan Sebastian Acuña Tellez jsacunat@udistrital.edu.co
Pablo Patiño Bonilla jppatinob@udistrital.edu.co


---

Good luck experimenting with PINNo.
