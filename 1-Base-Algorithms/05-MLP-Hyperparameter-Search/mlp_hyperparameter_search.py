"""Hyperparameter search (activation function x number of neurons) for an MLP.

Translated from: DESAFIO01_FUNCAO_MLP_keras.R and DESAFIO01_FUNCAO_MLP_neuralnet.R
"""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def load_data(path: str) -> pd.DataFrame:
    """Load the x/y dataset (with xgrd/ymean as the target function) from an Excel file."""
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
    """Train an MLP for every (activation, neurons, repeat) combination.

    Reproduces the original R scripts' experiment structure: for each
    configuration, several models are trained with different random
    initialization, and RMSE/R2 are recorded to assess both average
    performance and stability (variance) across configurations.
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
    """Aggregate the experiment by configuration: mean RMSE/R2 and RMSE std dev."""
    summary = (
        results.groupby(["activation", "hidden_size"])
        .agg(mean_rmse=("rmse", "mean"), rmse_std_dev=("rmse", "std"), mean_r2=("r_squared", "mean"))
        .reset_index()
        .sort_values("mean_rmse")
    )
    return summary


if __name__ == "__main__":
    dt = load_data("data/funcao.xlsx")
    x, y = dt["x"].to_numpy(dtype=float), dt["y"].to_numpy(dtype=float)

    results = run_experiment(
        x, y, activations=["relu", "tanh", "logistic"], hidden_sizes=[5, 10, 20], n_repeats=5
    )
    print(summarize(results).to_string(index=False))
