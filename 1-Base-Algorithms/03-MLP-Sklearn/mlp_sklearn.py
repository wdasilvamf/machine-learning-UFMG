"""MLP fitted with scikit-learn, equivalent to R's `neuralnet` package.

Translated from: Codigo01_Seno_MLP_neuralnet.R and Codigo01_FUNCAO_MLP_neuralnet.R
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


def load_data(path: str) -> pd.DataFrame:
    """Load an x/y dataset (optionally with xgrd/ymean) from an Excel file."""
    return pd.read_excel(path)


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (6,),
    activation: str = "tanh",
    max_iter: int = 5000,
    random_state: int | None = 42,
) -> MLPRegressor:
    """Fit a single-hidden-layer MLPRegressor, equivalent to `neuralnet(hidden=c(6))`.

    `solver="lbfgs"` is used (instead of the default "adam") because it
    converges more stably and deterministically on small datasets like this
    one, approximating the behavior of the optimizer used by the R
    `neuralnet` package.
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
    """Generate predictions from the trained model for an input vector x."""
    return model.predict(x.reshape(-1, 1))


if __name__ == "__main__":
    dt = load_data("data/Seno.xlsx")
    x, y = dt["x"].to_numpy(dtype=float), dt["y"].to_numpy(dtype=float)

    model = fit_mlp(x, y, hidden_layer_sizes=(6,), activation="tanh")
    y_hat = predict(model, x)
    print(f"R2 (train): {model.score(x.reshape(-1, 1), y):.4f}")
