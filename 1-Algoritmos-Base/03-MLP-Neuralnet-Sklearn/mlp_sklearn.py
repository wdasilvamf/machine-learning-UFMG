"""MLP ajustado com scikit-learn, equivalente ao pacote `neuralnet` do R.

Traduzido de: Codigo01_Seno_MLP_neuralnet.R e Codigo01_FUNCAO_MLP_neuralnet.R
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def load_data(path: str) -> pd.DataFrame:
    """Carrega um dataset x/y (e opcionalmente xgrd/ymean) de um arquivo Excel."""
    return pd.read_excel(path)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (6,),
    activation: str = "tanh",
    max_iter: int = 5000,
    random_state: int | None = 42,
) -> MLPRegressor:
    """Ajusta um MLPRegressor de camada unica, equivalente a `neuralnet(hidden=c(6))`.

    `solver="lbfgs"` e usado (em vez do default "adam") porque converge de
    forma mais estavel e deterministica em datasets pequenos como este,
    aproximando o comportamento do otimizador usado pelo pacote `neuralnet`.
    """
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(x.reshape(-1, 1), y)
    return model


def predict(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    """Gera predicoes do modelo treinado para um vetor de entradas x."""
    return model.predict(x.reshape(-1, 1))


if __name__ == "__main__":
    dt = load_data("data/Seno.xlsx")
    x, y = dt["x"].to_numpy(dtype=float), dt["y"].to_numpy(dtype=float)

    model = fit_mlp(x, y, hidden_layer_sizes=(6,), activation="tanh")
    y_hat = predict(model, x)
    print(f"R2 (train): {model.score(x.reshape(-1, 1), y):.4f}")
