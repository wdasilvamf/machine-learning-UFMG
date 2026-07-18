"""Multi-Layer Perceptron (1 camada escondida) com backpropagation implementado manualmente.

Traduzido de: Codigo01_Seno_MLP_Backpropagation.R
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MLPWeights:
    w1: np.ndarray  # (n_inputs, n_hidden) - pesos entrada -> camada escondida
    w2: np.ndarray  # (n_hidden, 1)        - pesos camada escondida -> saida
    b2: float        # bias da camada de saida


@dataclass
class TrainResult:
    weights: MLPWeights
    loss_history: np.ndarray


def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset x/y (seno ruidoso) de um arquivo Excel."""
    return pd.read_excel(path)


def build_design_matrix(x: np.ndarray) -> np.ndarray:
    """Adiciona a coluna de bias (intercepto) ao vetor de entrada."""
    return np.column_stack([np.ones_like(x), x])


def tanh_activation(z: np.ndarray) -> np.ndarray:
    """Tangente hiperbolica, escrita na mesma forma do script original em R."""
    return 2 / (1 + np.exp(-2 * z)) - 1


def tanh_derivative(activated: np.ndarray) -> np.ndarray:
    """Derivada de tanh em funcao da propria ativacao: d/dz tanh(z) = 1 - tanh(z)^2."""
    return 1 - activated**2


def init_weights(n_inputs: int, n_hidden: int, y_mean: float, seed: int | None = None) -> MLPWeights:
    """Inicializa os pesos com ruido gaussiano pequeno, como no script original."""
    rng = np.random.default_rng(seed)
    w1 = 0.01 * rng.standard_normal((n_inputs, n_hidden))
    w2 = 0.01 * rng.standard_normal((n_hidden, 1))
    return MLPWeights(w1=w1, w2=w2, b2=y_mean)


def forward(X: np.ndarray, weights: MLPWeights) -> tuple[np.ndarray, np.ndarray]:
    """Propagacao direta: retorna a ativacao da camada escondida e a saida da rede."""
    hidden = tanh_activation(X @ weights.w1)
    output = hidden @ weights.w2 + weights.b2
    return hidden, output


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int = 6,
    learning_rate: float = 0.01,
    n_steps: int = 100_000,
    seed: int | None = 42,
) -> TrainResult:
    """Treina o MLP via backpropagation (batch gradient descent).

    Reproduz o algoritmo do script original em R: uma camada escondida com
    ativacao tanh e saida linear, com gradientes calculados na forma matricial.
    """
    n_inputs = X.shape[1]
    n_train = X.shape[0]
    weights = init_weights(n_inputs, n_hidden, y_mean=float(y.mean()), seed=seed)
    loss_history = np.empty(n_steps)

    for step in range(n_steps):
        hidden, output = forward(X, weights)
        error = y - output

        grad_w2 = -hidden.T @ error
        grad_b2 = -np.sum(error)
        hidden_error = (error @ weights.w2.T) * tanh_derivative(hidden)
        grad_w1 = -X.T @ hidden_error

        weights.w1 -= learning_rate * (grad_w1 / n_train)
        weights.w2 -= learning_rate * (grad_w2 / n_train)
        weights.b2 -= learning_rate * (grad_b2 / n_train)

        _, output = forward(X, weights)
        loss_history[step] = np.sum((y - output) ** 2)

    return TrainResult(weights=weights, loss_history=loss_history)


if __name__ == "__main__":
    dt = load_data("data/Seno.xlsx")
    X = build_design_matrix(dt["x"].to_numpy(dtype=float))
    y = dt["y"].to_numpy(dtype=float).reshape(-1, 1)

    result = train_mlp(X, y, n_hidden=6, learning_rate=0.01, n_steps=20_000)
    print(f"SSE final: {result.loss_history[-1]:.4f}")
