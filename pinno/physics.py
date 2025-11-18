"""
Módulo que define los problemas físicos y sus ecuaciones diferenciales.

Este módulo contiene la lógica central de la "Física" en las PINNs. Define las clases
que encapsulan las Ecuaciones Diferenciales (ODEs y PDEs), calcula los residuos
utilizando diferenciación automática y proporciona soluciones analíticas para
validación.

Classes:
    PhysicsProblem: Clase base abstracta.
    SimpleHarmonicOscillator: Implementación del oscilador armónico (Masa-Resorte).
    DampedHarmonicOscillator: Implementación del oscilador amortiguado.
    HeatEquation2D: Implementación de la ecuación de calor en 2D.
"""

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

# ==============================================================================
# --- CLASE BASE ---
# ==============================================================================

class PhysicsProblem(ABC):
    """
    Clase base abstracta para todos los problemas físicos.

    Define la interfaz que deben implementar los problemas específicos para ser
    compatibles con el entrenador (PINNTrainer).

    Attributes:
        config (Dict[str, Any]): Configuración global del experimento.
        domain_config (dict): Sub-sección 'PHYSICS_CONFIG' con parámetros específicos
                              (ej. omega, zeta, alpha, dominios).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el problema físico.

        Args:
            config (Dict[str, Any]): Configuración completa proveniente de `config.py`.
        """
        self.config = config
        self.domain_config = config['PHYSICS_CONFIG']

    @abstractmethod
    def pde_residual(self, model: tf.keras.Model, points: tf.Tensor) -> tf.Tensor:
        """
        Calcula el residual de la ecuación diferencial en los puntos dados.

        El residual es la diferencia entre el lado izquierdo y derecho de la ecuación.
        Si la red neuronal respeta la física, este valor debe ser cercano a 0.

        Args:
            model (tf.keras.Model): La red neuronal que aproxima la solución.
            points (tf.Tensor): Puntos de colocación (inputs) donde evaluar la ecuación.

        Returns:
            tf.Tensor: Tensor con los valores del residual para cada punto.
        """
        pass
    
    @abstractmethod
    def analytical_solution(self, points: np.ndarray | tf.Tensor) -> np.ndarray:
        """
        Calcula la solución exacta (analítica) del problema.

        Se utiliza para calcular el error real y generar gráficas comparativas.

        Args:
            points (Union[np.ndarray, tf.Tensor]): Puntos donde evaluar la solución.

        Returns:
            np.ndarray: Valores exactos de la solución.
        """
        pass


# ==============================================================================
# --- IMPLEMENTACIONES ESPECÍFICAS ---
# ==============================================================================

class SimpleHarmonicOscillator(PhysicsProblem):
    """
    Oscilador Armónico Simple (SHO).

    Resuelve la Ecuación Diferencial Ordinaria (ODE) de segundo orden:
        x''(t) + omega^2 * x(t) = 0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.omega = tf.constant(self.domain_config['omega'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, t: tf.Tensor) -> tf.Tensor:
        """
        Calcula el residual para SHO: R = x_tt + omega^2 * x.
        
        Usa `tf.GradientTape` anidado/persistente para calcular derivadas de segundo orden.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            x_t = tape.gradient(x, t) # Primera derivada (velocidad)
            
        x_tt = tape.gradient(x_t, t)  # Segunda derivada (aceleración)
        del tape
        
        if x_tt is None:
            raise ValueError("El cálculo de la segunda derivada (x_tt) falló. "
                             "Verifique que la función de activación sea diferenciable.")
                             
        return x_tt + (self.omega**2) * x
    
    def analytical_solution(self, t) -> np.ndarray:
        """
        Solución: x(t) = x0*cos(wt) + (v0/w)*sin(wt).
        """
        t_val = t.numpy() if hasattr(t, 'numpy') else t
        x0 = self.domain_config['initial_conditions']['x0']
        v0 = self.domain_config['initial_conditions']['v0']
        omega_val = self.omega.numpy()
        return x0 * np.cos(omega_val * t_val) + (v0 / omega_val) * np.sin(omega_val * t_val)


class DampedHarmonicOscillator(SimpleHarmonicOscillator):
    """
    Oscilador Armónico Amortiguado (DHO).

    Resuelve la ODE con término de fricción:
        x''(t) + 2*zeta*omega * x'(t) + omega^2 * x(t) = 0
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.zeta = tf.constant(self.domain_config['zeta'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, t: tf.Tensor) -> tf.Tensor:
        """
        Calcula el residual para DHO. Requiere x (posición), x_t (velocidad) y x_tt (aceleración).
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x = model(t)
            x_t = tape.gradient(x, t)
        x_tt = tape.gradient(x_t, t)
        del tape

        if x_tt is None or x_t is None:
             raise ValueError("El cálculo de derivadas para DHO falló.")

        # Ecuación: x'' + 2*zeta*omega*x' + omega^2*x = 0
        return x_tt + 2 * self.zeta * self.omega * x_t + (self.omega**2) * x

    def analytical_solution(self, t) -> np.ndarray:
        """
        Solución para caso subamortiguado (zeta < 1):
        x(t) = exp(-zeta*w*t) * [A*cos(wd*t) + B*sin(wd*t)]
        """
        t_val = t.numpy() if hasattr(t, 'numpy') else t
        x0 = self.domain_config['initial_conditions']['x0']
        v0 = self.domain_config['initial_conditions']['v0']
        omega_val = self.omega.numpy()
        zeta_val = self.zeta.numpy()
        
        if zeta_val < 1:
            omega_d = omega_val * np.sqrt(1 - zeta_val**2)
            A = x0
            B = (v0 + zeta_val * omega_val * x0) / omega_d
            return np.exp(-zeta_val * omega_val * t_val) * (A * np.cos(omega_d * t_val) + B * np.sin(omega_d * t_val))
        else:
            # Implementación simplificada: retorna ceros si no es subamortiguado
            return np.zeros_like(t_val)


class HeatEquation2D(PhysicsProblem):
    """
    Ecuación de Calor en 2D (Dependiente del tiempo).

    Resuelve la Ecuación Diferencial Parcial (PDE):
        u_t - alpha * (u_xx + u_yy) = 0
        
    Donde 'u' es la temperatura, 't' es tiempo, 'x,y' espacio y 'alpha' la difusividad.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha = tf.constant(self.domain_config['alpha'], dtype=tf.float32)

    def pde_residual(self, model: tf.keras.Model, xyt: tf.Tensor) -> tf.Tensor:
        """
        Calcula el residual de la PDE de calor.
        
        Args:
            xyt: Tensor de forma (N, 3) conteniendo columnas [x, y, t].
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xyt)
            u = model(xyt)
            # Primeras derivadas (Jacobiano)
            # Slicing: [:, 0:1] es du/dx, [:, 1:2] es du/dy, [:, 2:3] es du/dt
            u_x = tape.gradient(u, xyt)[:, 0:1]
            u_y = tape.gradient(u, xyt)[:, 1:2]
            u_t = tape.gradient(u, xyt)[:, 2:3]
        
        # Segundas derivadas (Hessiano diagonal)
        u_xx = tape.gradient(u_x, xyt)[:, 0:1]
        u_yy = tape.gradient(u_y, xyt)[:, 1:2]
        del tape

        if u_xx is None or u_yy is None or u_t is None:
            raise ValueError("El cálculo de derivadas para HEAT falló.")

        return u_t - self.alpha * (u_xx + u_yy)

    def analytical_solution(self, xyt) -> np.ndarray:
        """
        Solución analítica particular para una placa cuadrada con condiciones iniciales sinusoidales.
        u(x,y,t) = exp(-alpha * pi^2 * t) * sin(pi*x) * sin(pi*y)
        """
        xyt_val = xyt.numpy() if hasattr(xyt, 'numpy') else xyt
        x = xyt_val[:, 0:1]
        y = xyt_val[:, 1:2] 
        t = xyt_val[:, 2:3]
        alpha_val = self.alpha.numpy()
        return np.exp(-alpha_val * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def compute_slice_metrics(self, model: tf.keras.Model, t_fix: float = 0.5, resolution: int = 50) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calcula métricas y datos para un corte de tiempo específico (Snapshot).

        Útil para visualizar el campo de temperatura en un instante 't' fijo.

        Args:
            model: Modelo entrenado.
            t_fix (float): Instante de tiempo a evaluar.
            resolution (int): Resolución de la malla espacial (N x N).

        Returns:
            Tuple: 
                - mse (float): Error cuadrático medio en este corte.
                - u_pred (ndarray): Matriz (N, N) con predicciones.
                - u_true (ndarray): Matriz (N, N) con valores reales.
        """
        x_d = self.domain_config['x_domain']
        y_d = self.domain_config['y_domain']
        
        # Crear grid espacial
        x = np.linspace(x_d[0], x_d[1], resolution)
        y = np.linspace(y_d[0], y_d[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Crear tensor input con t fijo
        T = np.full_like(X, t_fix)
        
        # Aplanar y apilar: (N*N, 3) -> [x, y, t]
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = T.flatten()
        
        input_array = np.stack([x_flat, y_flat, t_flat], axis=1).astype(np.float32)
        input_tensor = tf.convert_to_tensor(input_array)
        
        # Predicción
        u_pred_flat = model(input_tensor).numpy()
        u_true_flat = self.analytical_solution(input_tensor)
        
        # Calcular MSE
        mse = np.mean((u_pred_flat - u_true_flat)**2)
        
        # Reshape para visualización (N, N)
        return mse, u_pred_flat.reshape(resolution, resolution), u_true_flat.reshape(resolution, resolution)


# ==============================================================================
# --- FÁBRICA DE PROBLEMAS ---
# ==============================================================================

PROBLEMS: Dict[str, type] = {
    "SHO": SimpleHarmonicOscillator,
    "DHO": DampedHarmonicOscillator,
    "HEAT": HeatEquation2D
}
"""dict: Registro de clases de problemas físicos disponibles."""

def get_physics_problem(problem_name: str, config: Dict[str, Any]) -> PhysicsProblem:
    """
    Fábrica que instancia un problema físico basado en su nombre.

    Args:
        problem_name (str): Identificador ("SHO", "HEAT", etc.).
        config (Dict[str, Any]): Configuración global a inyectar en la instancia.

    Returns:
        PhysicsProblem: Instancia configurada del problema solicitado.

    Raises:
        ValueError: Si el problema no existe en el registro `PROBLEMS`.
    """
    problem_name = problem_name.upper()
    if problem_name not in PROBLEMS:
        raise ValueError(f"Problema '{problem_name}' no reconocido.")
    return PROBLEMS[problem_name](config)