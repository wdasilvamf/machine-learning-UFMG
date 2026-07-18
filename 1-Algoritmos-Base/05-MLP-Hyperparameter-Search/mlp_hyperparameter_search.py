"""Busca de hiperparametros (funcao de ativacao x numero de neuronios) para um MLP.

Traduzido de: DESAFIO01_FUNCAO_MLP_keras.R e DESAFIO01_FUNCAO_MLP_neuralnet.R
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset x/y (com xgrd/ymean como funcao alvo) de um arquivo Excel."""
    return pd.read_excel(path)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def run_experiment(
    x: np.ndarray,
    y: np.ndarray,
    activations: list[str],
    hidden_sizes: list[int],
    n_repeats: int = 5,
    max_iter: int = 2000,
    random_state_base: int = 0,
) -> pd.DataFrame:
    """Treina um MLP para cada combinacao (ativacao, neuronios, repeticao).

    Reproduz a estrutura de experimento dos scripts originais em R: para cada
    configuracao, varios modelos sao treinados com inicializacao aleatoria
    diferente, e RMSE/R2 sao registrados para avaliar tanto o desempenho
    medio quanto a estabilidade (variancia) de cada configuracao.
    """
    rows = []
    x_2d = x.reshape(-1, 1)

    for activation, hidden_size, repeat in product(activations, hidden_sizes, range(n_repeats)):
        seed = random_state_base + repeat
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_size,),
            activation=activation,
            solver="adam",
            max_iter=max_iter,
            random_state=seed,
        )
        model.fit(x_2d, y)
        y_pred = model.predict(x_2d)

        rows.append(
            {
                "activation": activation,
                "hidden_size": hidden_size,
                "repeat": repeat,
                "rmse": rmse(y, y_pred),
                "r_squared": r_squared(y, y_pred),
                "y_pred": y_pred,
            }
        )

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Agrega o experimento por configuracao: RMSE/R2 medios e desvio-padrao do RMSE."""
    summary = (
        results.groupby(["activation", "hidden_size"])
        .agg(rmse_medio=("rmse", "mean"), rmse_desvio_padrao=("rmse", "std"), r2_medio=("r_squared", "mean"))
        .reset_index()
        .sort_values("rmse_medio")
    )
    return summary


if __name__ == "__main__":
    dt = load_data("data/funcao.xlsx")
    x, y = dt["x"].to_numpy(dtype=float), dt["y"].to_numpy(dtype=float)

    results = run_experiment(
        x, y, activations=["relu", "tanh", "logistic"], hidden_sizes=[5, 10, 20], n_repeats=5
    )
    print(summarize(results).drop(columns=[]).to_string(index=False))
