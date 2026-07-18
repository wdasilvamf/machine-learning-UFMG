"""Simple linear regression via gradient descent, compared to the closed-form (OLS) solution.

Translated from: Codigo01_SalarioExperiencia_BackPropagation.R
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GradientDescentResult:
    intercept: float
    slope: float
    sse_history: np.ndarray


def load_data(path: str) -> pd.DataFrame:
    """Load the Salary x Experience dataset from an Excel file."""
    return pd.read_excel(path)


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit simple linear regression via the closed-form least squares solution."""
    x_design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return intercept, slope


def fit_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.0001,
    n_steps: int = 10_000,
) -> GradientDescentResult:
    """Fit simple linear regression via (batch) gradient descent.

    Reproduces the manual backpropagation algorithm from the original R
    script: at each step, the coefficients are updated in the direction
    opposite to the gradient of the sum of squared errors (SSE).
    """
    intercept, slope = 0.0, 0.0
    sse_history = np.empty(n_steps)

    for step in range(n_steps):
        residual = y - (intercept + slope * x)
        grad_intercept = -np.sum(residual)
        grad_slope = -np.sum(residual * x)

        intercept -= learning_rate * grad_intercept
        slope -= learning_rate * grad_slope

        y_hat = intercept + slope * x
        sse_history[step] = np.sum((y - y_hat) ** 2)

    return GradientDescentResult(intercept=intercept, slope=slope, sse_history=sse_history)


if __name__ == "__main__":
    dt = load_data("data/SalarioExperiencia.xlsx")
    x, y = dt["Experiencia"].to_numpy(dtype=float), dt["Salario"].to_numpy(dtype=float)

    ols_intercept, ols_slope = fit_ols(x, y)
    gd_result = fit_gradient_descent(x, y)

    print(f"Least squares    : intercept={ols_intercept:.4f}, slope={ols_slope:.4f}")
    print(f"Gradient descent : intercept={gd_result.intercept:.4f}, slope={gd_result.slope:.4f}")
