"""
Interfaz web (Streamlit) para PINN Interactive Trainer.

Este módulo define la aplicación web completa utilizando el framework Streamlit.
A diferencia de la versión de escritorio, esta aplicación se ejecuta en el navegador
y se basa en un modelo de ejecución reactivo (el script se re-ejecuta con cada interacción).

Características principales:
- **Path Patching**: Configuración dinámica de `sys.path` para permitir importaciones
  absolutas del paquete `pinno` sin necesidad de instalación previa.
- **Gestión de Estado**: Uso de `st.session_state` para persistir el modelo y
  el historial de entrenamiento entre re-ejecuciones del script.
- **Visualización Reactiva**: Gráficos que se actualizan en tiempo real durante
  el bucle de entrenamiento.

Usage:
    Para ejecutar esta aplicación, utilice el comando:
    $ streamlit run src/gui_web.py
    
    O si está instalado como paquete:
    $ streamlit run path/to/site-packages/pinno/gui_web.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import json
import os
import sys
import tensorflow as tf

# ==============================================================================
# --- CONFIGURACIÓN DE ENTORNO E IMPORTACIONES ---
# ==============================================================================

# --- 🛠️ CORRECCIÓN DE RUTAS (Path Patching) ---
# Streamlit ejecuta los scripts como archivos independientes, no como módulos.
# Esto rompe las importaciones relativas (from . import config).
# Solución: Agregamos la carpeta raíz del proyecto al PYTHONPATH dinámicamente.

current_dir = os.path.dirname(os.path.abspath(__file__)) # Directorio actual (pinno/)
root_dir = os.path.dirname(current_dir)                  # Directorio padre (PINNo/)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- IMPORTACIONES DEL NÚCLEO ---
try:
    # Intento 1: Importación relativa (funciona si se lanza con python -m)
    from .config import get_active_config
    from .training import PINNTrainer
    from .models import get_model
except ImportError:
    # Intento 2: Importación absoluta (funciona con Path Patching o si está instalado)
    # Nota: Asumimos que la carpeta del paquete se llama 'pinno' o 'src'
    try:
        from pinno.config import get_active_config
        from pinno.training import PINNTrainer
        from pinno.models import get_model
    except ImportError:
        # Fallback para estructura antigua 'src'
        from src.config import get_active_config
        from src.training import PINNTrainer
        from src.models import get_model

# ==============================================================================
# --- CONFIGURACIÓN DE LA INTERFAZ DE USUARIO ---
# ==============================================================================

st.set_page_config(
    page_title="PINNo - Interactive Trainer",
    page_icon="🧠",
    layout="wide"
)

# CSS personalizado para mejorar la estética de botones y métricas
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# --- GESTIÓN DEL ESTADO (SESSION STATE) ---
# ==============================================================================
# Inicializamos variables que deben persistir entre recargas de página

if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = {"loss": [], "epochs": []}
if 'external_data' not in st.session_state:
    st.session_state.external_data = None
if 'current_problem' not in st.session_state:
    st.session_state.current_problem = "SHO"


def reset_training(problem_name: str):
    """
    Reinicia el estado del entrenamiento para comenzar un nuevo experimento.

    Limpia el historial de pérdidas, detiene cualquier entrenamiento activo
    y reinicia la instancia del entrenador.

    Args:
        problem_name (str): Identificador del nuevo problema físico a resolver
                            (ej. "SHO", "HEAT").
    """
    st.session_state.training_history = {"loss": [], "epochs": []}
    st.session_state.is_training = False
    st.session_state.trainer = None
    st.session_state.current_problem = problem_name


# ==============================================================================
# --- INTERFAZ PRINCIPAL ---
# ==============================================================================

st.title("🧠 PINNo: Physics-Informed Neural Networks")
st.markdown("### Plataforma Interactiva de Entrenamiento y Visualización")

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Learn PINNs", 
    "📂 Data Explorer", 
    "⚙️ Train & Config", 
    "📊 Metrics Report"
])

# --- PESTAÑA 1: EDUCACIÓN ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("¿Qué es una PINN?")
        st.info("""
        Una **Red Neuronal Informada por la Física (PINN)** es un modelo de Deep Learning que no solo aprende de datos, 
        sino que también respeta las leyes de la física descritas por Ecuaciones Diferenciales (EDP).
        """)
        st.markdown("#### ¿Cómo funciona?")
        st.markdown(r"""
        A diferencia de una red normal ("Caja Negra"), una PINN optimiza una función de pérdida compuesta:
        $$ L_{total} = w_{data} \cdot L_{datos} + w_{physics} \cdot L_{física} $$
        Donde $L_{física}$ es el **residual** de la ecuación diferencial.
        """)
    with col2:
        st.subheader("Conceptos Clave")
        with st.expander("Epochs (Épocas)"):
            st.write("Cuántas veces la red revisa el problema completo.")
        with st.expander("Learning Rate"):
            st.write("El tamaño del paso al corregir errores.")

# --- PESTAÑA 2: EXPLORADOR DE DATOS ---
with tab2:
    st.header("Carga de Datos Experimentales")
    uploaded_file = st.file_uploader("Arrastra tu archivo aquí (.csv / .xlsx)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.external_data = df
            st.success(f"✅ Archivo cargado: {len(df)} filas.")
            
            c1, c2 = st.columns([1, 2])
            with c1: st.dataframe(df.head(10))
            with c2:
                cols = df.columns
                if len(cols) >= 2: st.line_chart(df.set_index(cols[0])[cols[1]])
        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")

# --- PESTAÑA 3: ENTRENAMIENTO ---
with tab3:
    col_config, col_viz = st.columns([1, 2])
    
    with col_config:
        st.subheader("Configuración")
        problem = st.selectbox("Problema Físico", ["SHO", "DHO", "HEAT"])
        base_config = get_active_config(problem)
        
        with st.expander("🧠 Arquitectura", expanded=True):
            num_layers = st.number_input("Capas", 2, 10, base_config["MODEL_CONFIG"]["num_layers"])
            hidden_dim = st.number_input("Neuronas", 16, 256, base_config["MODEL_CONFIG"]["hidden_dim"])
            activation = st.selectbox("Activación", ["tanh", "relu", "sigmoid", "swish"], index=0)
        
        with st.expander("⚙️ Hiperparámetros"):
            lr = st.number_input("Learning Rate", 1e-5, 1e-1, base_config["LEARNING_RATE"], format="%.5f")
            epochs_target = st.number_input("Epochs", 100, 50000, base_config["EPOCHS"])
        
        with st.expander("⚛️ Física"):
            phys_params = {}
            for k, v in base_config["PHYSICS_CONFIG"].items():
                if isinstance(v, (int, float)):
                    phys_params[k] = st.number_input(f"{k}", value=float(v))
        
        start_btn = st.button("🚀 Iniciar", type="primary")
        stop_btn = st.button("🛑 Detener")
        
        if start_btn:
            reset_training(problem)
            base_config["MODEL_CONFIG"].update({"num_layers": num_layers, "hidden_dim": hidden_dim, "activation": activation})
            base_config["LEARNING_RATE"] = lr
            base_config["EPOCHS"] = epochs_target
            for k, v in phys_params.items(): base_config["PHYSICS_CONFIG"][k] = v
            
            st.session_state.trainer = PINNTrainer(base_config, problem)
            st.session_state.is_training = True

        if stop_btn:
            st.session_state.is_training = False

    with col_viz:
        st.subheader("Monitor en Tiempo Real")
        loss_chart = st.empty()
        solution_chart = st.empty()
        metric_box = st.empty()
        
        if st.session_state.is_training:
            progress_bar = st.progress(0)
            trainer = st.session_state.trainer
            
            while st.session_state.is_training and trainer.epoch < epochs_target:
                current_loss = 0
                for _ in range(50): 
                    losses = trainer.perform_one_step()
                    current_loss = losses[0].numpy()
                    st.session_state.training_history["loss"].append(current_loss)
                
                metric_box.markdown(f"**Epoch:** {trainer.epoch} | **Loss:** {current_loss:.6f}")
                progress_bar.progress(min(trainer.epoch / epochs_target, 1.0))
                
                loss_df = pd.DataFrame(st.session_state.training_history["loss"], columns=["Loss"])
                loss_chart.line_chart(loss_df, height=200)
                
                # --- VISUALIZACIÓN ADAPTATIVA ---
                fig, ax = plt.subplots(figsize=(8, 4))
                
                if problem in ["SHO", "DHO"]:
                    t_d = base_config["PHYSICS_CONFIG"]["t_domain"]
                    t_test = np.linspace(t_d[0], t_d[1], 100).reshape(-1,1).astype(np.float32)
                    u_p = trainer.model(t_test).numpy()
                    u_t = trainer.physics.analytical_solution(t_test)
                    ax.plot(t_test, u_t, 'k--', label="Real")
                    ax.plot(t_test, u_p, 'r-', label="PINN")
                    ax.set_title(f"Dinámica {problem}")
                    ax.legend()
                    solution_chart.pyplot(fig)
                    
                elif problem == "HEAT":
                    N = 40 
                    x_d = base_config["PHYSICS_CONFIG"]["x_domain"]
                    y_d = base_config["PHYSICS_CONFIG"]["y_domain"]
                    x = np.linspace(x_d[0], x_d[1], N)
                    y = np.linspace(y_d[0], y_d[1], N)
                    X, Y = np.meshgrid(x, y)
                    t_mid = 0.5
                    
                    inp = tf.convert_to_tensor(np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), t_mid)], axis=1).astype(np.float32))
                    Z_pred = trainer.model(inp).numpy().reshape(N, N)
                    Z_true = trainer.physics.analytical_solution(inp).reshape(N, N)
                    
                    plt.close(fig)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    vmin = min(Z_true.min(), Z_pred.min())
                    vmax = max(Z_true.max(), Z_pred.max())
                    
                    im1 = ax1.imshow(Z_pred, extent=[x_d[0], x_d[1], y_d[0], y_d[1]], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                    ax1.set_title(f"Predicción t={t_mid}")
                    im2 = ax2.imshow(Z_true, extent=[x_d[0], x_d[1], y_d[0], y_d[1]], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                    ax2.set_title(f"Analítica t={t_mid}")
                    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
                    solution_chart.pyplot(fig)
                
                plt.close(fig)
                if not st.session_state.is_training: break

# --- PESTAÑA 4: REPORTES ---
with tab4:
    st.header("Resultados y Exportación")
    
    if not st.session_state.training_history["loss"]:
        st.warning("No hay datos. Entrena un modelo primero.")
    else:
        hist = st.session_state.training_history["loss"]
        trainer = st.session_state.trainer
        problem = st.session_state.current_problem

        c1, c2, c3 = st.columns(3)
        c1.metric("Loss Final", f"{hist[-1]:.2e}")
        c2.metric("Mejor Loss", f"{min(hist):.2e}")
        c3.metric("Problema", problem)
        
        st.divider()
        col_graphs, col_actions = st.columns([3, 1])
        
        with col_graphs:
            st.subheader("📈 Convergencia")
            fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
            ax_loss.plot(hist, label="Total Loss", color="#1f77b4")
            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Epochs")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.3)
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            if problem == "HEAT" and trainer:
                st.subheader("🔥 Mapa de Calor Final")
                N = 100 
                x_d = trainer.config["PHYSICS_CONFIG"]["x_domain"]
                y_d = trainer.config["PHYSICS_CONFIG"]["y_domain"]
                x = np.linspace(x_d[0], x_d[1], N)
                y = np.linspace(y_d[0], y_d[1], N)
                X, Y = np.meshgrid(x, y)
                inp = tf.convert_to_tensor(np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), 0.5)], axis=1).astype(np.float32))
                Z_pred = trainer.model(inp).numpy().reshape(N, N)
                Z_true = trainer.physics.analytical_solution(inp).reshape(N, N)
                
                fig_hm, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                vmin, vmax = min(Z_true.min(), Z_pred.min()), max(Z_true.max(), Z_pred.max())
                ax1.imshow(Z_pred, extent=[0,1,0,1], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                ax1.set_title("Predicción")
                im = ax2.imshow(Z_true, extent=[0,1,0,1], origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                ax2.set_title("Analítica")
                fig_hm.colorbar(im, ax=[ax1, ax2], fraction=0.046, pad=0.04)
                st.pyplot(fig_hm)
                plt.close(fig_hm)

        with col_actions:
            st.subheader("📥 Exportar")
            csv = pd.DataFrame(hist, columns=["Loss"]).to_csv(index=False).encode('utf-8')
            st.download_button("📄 CSV Historial", data=csv, file_name="history.csv", mime="text/csv")
            
            if trainer:
                model_path = "temp_model.keras"
                trainer.model.save(model_path)
                with open(model_path, "rb") as f:
                    st.download_button("🧠 Modelo (.keras)", data=f, file_name="pinn_model.keras")