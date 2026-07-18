"""Polynomial regression with and without L2 (Ridge) regularization.

Translated from: Codigo01_Seno_Regularizacao.R
"""

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the x/y dataset (noisy sine) from an Excel file."""
    return pd.read_excel(path)


def build_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Build the design matrix [1, x, x^2, ..., x^degree] (raw basis, not orthogonalized).

    Equivalent to `model.matrix(lm(y ~ poly(x, degree, raw = TRUE)))` in R.
    """
    return np.column_stack([x**d for d in range(degree + 1)])


def fit_polynomial_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit the coefficients via ordinary least squares (no regularization)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float, penalize_intercept: bool = False) -> np.ndarray:
    """Fit the coefficients via regularized (Ridge/L2) least squares.

    beta = (X'X + lambda*I)^-1 X'y. By default the intercept (first column
    of X) is excluded from the penalty — same convention as the original R
    script. `penalize_intercept=True` is exposed only to demonstrate why
    that convention matters (see the notebook).
    """
    n_features = X.shape[1]
    penalty = np.eye(n_features)
    if not penalize_intercept:
        penalty[0, 0] = 0
    return np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Apply fitted coefficients to a design matrix."""
    return X @ beta


if __name__ == "__main__":
    dt = load_data("data/Seno.xlsx")
    x, y = dt["x"].to_numpy(dtype=float), dt["y"].to_numpy(dtype=float)

    X = build_polynomial_features(x, degree=8)
    beta_ols = fit_polynomial_ols(X, y)
    beta_ridge = fit_ridge(X, y, lam=0.5)

    print(f"Coefficient norm (OLS)  : {np.linalg.norm(beta_ols[1:]):.4f}")
    print(f"Coefficient norm (Ridge): {np.linalg.norm(beta_ridge[1:]):.4f}")
