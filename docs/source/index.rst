.. PINNo documentation master file

Bienvenido a la documentación de PINNo
=============================================

**PINNo** (Physics-Informed Neural Network Interactive Trainer) es un framework educativo y de investigación diseñado para experimentar rápidamente con Redes Neuronales Informadas por la Física.

.. toctree::
   :maxdepth: 2
   :caption: Contenido:

   api_reference

Fundamentos Teóricos (PINNs)
============================

Las **Physics-Informed Neural Networks (PINNs)** son una clase de algoritmos de aprendizaje profundo capaces de resolver tareas supervisadas respetando las leyes físicas descritas por Ecuaciones Diferenciales Parciales (EDP).

La Función de Coste Compuesta
-----------------------------

A diferencia de las redes neuronales tradicionales que solo minimizan el error respecto a los datos, una PINN minimiza una función compuesta:

.. math::

   \mathcal{L}_{total} = \omega_{data} \mathcal{L}_{data} + \omega_{PDE} \mathcal{L}_{physics}

Donde:

* :math:`\mathcal{L}_{data}`: Error Cuadrático Medio (MSE) entre la predicción :math:`u` y los datos observados :math:`u_{obs}`.
* :math:`\mathcal{L}_{physics}`: El residuo cuadrático de la ecuación diferencial.

El Residuo Físico
-----------------

Si definimos una ecuación diferencial general como :math:`\mathcal{N}[u] = 0`, el residuo :math:`f` se define como:

.. math::

   f = \mathcal{N}[\hat{u}(t, x; \theta)]

Durante el entrenamiento, la red ajusta sus pesos :math:`\theta` para minimizar :math:`||f||^2`, forzando a la solución a cumplir la física.

Ejemplos de Uso
===============

PINNo permite dos modos de operación principales.

Modo 1: Analítico (Solo Física)
-------------------------------

Ideal para encontrar la solución de una ecuación sin tener datos experimentales, solo condiciones iniciales.

.. code-block:: python

   from pinno.config import get_active_config
   from pinno.training import PINNTrainer

   # 1. Cargar configuración para Oscilador Armónico
   config = get_active_config("SHO")
   config["EPOCHS"] = 5000

   # 2. Inicializar Entrenador
   trainer = PINNTrainer(config, "SHO")

   # 3. Entrenar
   print("Iniciando entrenamiento...")
   for _ in range(5000):
       trainer.perform_one_step()

Modo 2: Data-Driven (Híbrido)
-----------------------------

Usa datos de un CSV para guiar la solución, útil para problemas inversos o descubrimiento de parámetros.

.. code-block:: python

   import pandas as pd
   from pinno.training import PINNTrainer
   from pinno.config import get_active_config

   # 1. Cargar datos
   df = pd.read_csv("datos_sensor.csv")
   
   # 2. Mapear columnas del CSV a variables físicas
   mapping = {
       "time": "t_sec",
       "displacement": "x_pos"
   }

   # 3. Entrenar (El sistema detecta el modo híbrido automáticamente)
   trainer = PINNTrainer(
       get_active_config("SHO"), 
       "SHO", 
       csv_data=df, 
       column_mapping=mapping
   )

Índices y Tablas
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
