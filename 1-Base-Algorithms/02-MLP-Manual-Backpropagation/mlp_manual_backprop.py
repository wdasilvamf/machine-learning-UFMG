"""Multi-Layer Perceptron (1 hidden layer) with manually implemented backpropagation.

Translated from: Codigo01_Seno_MLP_Backpropagation.R
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class MLPWeights:
    w1: np.ndarray  # (n_inputs, n_hidden) - input -> hidden layer weights
    w2: np.ndarray  # (n_hidden, 1)        - hidden layer -> output weights
    b2: float        # output layer bias


@dataclass
class TrainResult:
    weights: MLPWeights
    loss_history: np.ndarray


def load_data(path: str) -> pd.DataFrame:
    """Load the x/y dataset (noisy sine) from an Excel file."""
    return pd.read_excel(path)


def build_design_matrix(x: np.ndarray) -> np.ndarray:
    """Prepend the bias (intercept) column to the input vector."""
    return np.column_stack([np.ones_like(x), x])


def tanh_activation(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent, written in the same form as the original R script."""
    return 2 / (1 + np.exp(-2 * z)) - 1


def tanh_derivative(activated: np.ndarray) -> np.ndarray:
    """Derivative of tanh as a function of its own output: d/dz tanh(z) = 1 - tanh(z)^2."""
    return 1 - activated**2


def identity_activation(z: np.ndarray) -> np.ndarray:
    """Identity (no non-linearity) — used to demonstrate why activation functions matter."""
    return z


def identity_derivative(activated: np.ndarray) -> np.ndarray:
    """Derivative of the identity function: always 1."""
    return np.ones_like(activated)


def init_weights(n_inputs: int, n_hidden: int, y_mean: float, seed: int | None = None) -> MLPWeights:
    """Initialize weights with small Gaussian noise, as in the original script."""
    rng = np.random.default_rng(seed)
    w1 = 0.01 * rng.standard_normal((n_inputs, n_hidden))
    w2 = 0.01 * rng.standard_normal((n_hidden, 1))
    return MLPWeights(w1=w1, w2=w2, b2=y_mean)


def forward(
    X: np.ndarray, weights: MLPWeights, activation: Callable = tanh_activation
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass: returns the hidden layer's activation and the network's output."""
    hidden = activation(X @ weights.w1)
    output = hidden @ weights.w2 + weights.b2
    return hidden, output


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int = 6,
    learning_rate: float = 0.01,
    n_steps: int = 100_000,
    seed: int | None = 42,
    activation: Callable = tanh_activation,
    activation_derivative: Callable = tanh_derivative,
) -> TrainResult:
    """Train the MLP via backpropagation (batch gradient descent).

    Reproduces the algorithm from the original R script: one hidden layer
    with (by default) tanh activation and a linear output, with gradients
    computed in matrix form. `activation`/`activation_derivative` can be
    swapped (e.g. for `identity_activation`) to demonstrate what happens
    without a non-linearity.
    """
    n_inputs = X.shape[1]
    n_train = X.shape[0]
    weights = init_weights(n_inputs, n_hidden, y_mean=float(y.mean()), seed=seed)
    loss_history = np.empty(n_steps)

    for step in range(n_steps):
        hidden, output = forward(X, weights, activation)
        error = y - output

        grad_w2 = -hidden.T @ error
        grad_b2 = -np.sum(error)
        hidden_error = (error @ weights.w2.T) * activation_derivative(hidden)
        grad_w1 = -X.T @ hidden_error

        weights.w1 -= learning_rate * (grad_w1 / n_train)
        weights.w2 -= learning_rate * (grad_w2 / n_train)
        weights.b2 -= learning_rate * (grad_b2 / n_train)

        _, output = forward(X, weights, activation)
        loss_history[step] = np.sum((y - output) ** 2)

    return TrainResult(weights=weights, loss_history=loss_history)


if __name__ == "__main__":
    dt = load_data("data/Seno.xlsx")
    X = build_design_matrix(dt["x"].to_numpy(dtype=float))
    y = dt["y"].to_numpy(dtype=float).reshape(-1, 1)

    result = train_mlp(X, y, n_hidden=6, learning_rate=0.01, n_steps=20_000)
    print(f"Final SSE: {result.loss_history[-1]:.4f}")
