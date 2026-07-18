"""Regressao polinomial com e sem regularizacao L2 (Ridge).

Traduzido de: Codigo01_Seno_Regularizacao.R
"""

import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset x/y (seno ruidoso) de um arquivo Excel."""
    return pd.read_excel(path)


def build_polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Constroi a matriz de design [1, x, x^2, ..., x^degree] (base 'raw', sem ortogonalizar).

    Equivalente a `model.matrix(lm(y ~ poly(x, degree, raw = TRUE)))` no R.
    """
    return np.column_stack([x**d for d in range(degree + 1)])


def fit_polynomial_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ajusta os coeficientes por minimos quadrados ordinarios (sem regularizacao)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ajusta os coeficientes por minimos quadrados regularizados (Ridge/L2).

    beta = (X'X + lambda*I)^-1 X'y, com o intercepto (primeira coluna de X)
    excluido da penalizacao — mesma convencao usada no script original em R.
    """
    n_features = X.shape[1]
    penalty = np.eye(n_features)
    penalty[0, 0] = 0  # nao penaliza o intercepto
    return np.linalg.solve(X.T @ X + lam * penalty, X.T @ y)


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Aplica os coeficientes ajustados a uma matriz de design."""
    return X @ beta


if __name__ == "__main__":
    dt = load_data("data/Seno.xlsx")
    x, y = dt["x"].to_numpy(dtype=float), dt["y"].to_numpy(dtype=float)

    X = build_polynomial_features(x, degree=8)
    beta_ols = fit_polynomial_ols(X, y)
    beta_ridge = fit_ridge(X, y, lam=0.5)

    print(f"Norma dos coeficientes (OLS)  : {np.linalg.norm(beta_ols[1:]):.4f}")
    print(f"Norma dos coeficientes (Ridge): {np.linalg.norm(beta_ridge[1:]):.4f}")
