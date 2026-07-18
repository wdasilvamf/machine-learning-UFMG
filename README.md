# Machine Learning — UFMG
### Deep Learning Course | Prof. Marcelo A. Costa
**R → Python Translation & Portfolio** · [wdasilvamf](https://github.com/wdasilvamf)

---

Python translation of the R scripts from the Deep Learning course at UFMG, with **from-scratch** implementations (no high-level abstractions) to build a solid understanding of the math behind each model before moving on to frameworks.

The portfolio follows a deliberate progression: from linear models to MLPs, regularization, Keras, and eventually computer vision applied to my Master's research in Technological Innovation (Unifesp), focused on cognitive systems and social robotics.

## Structure

```
├── 1-Base-Algorithms/           # Faithful translation of the course scripts, one algorithm per folder
│   └── NN-Algorithm-Name/
│       ├── algorithm.py         # Modular implementation (functions + docstrings)
│       ├── algorithm.ipynb      # Theory, hyperparameters, and metric interpretation
│       └── data/                 # Original dataset from the R script
│
└── 2-Real-World-Kaggle-Cases/   # The same algorithms applied to real Kaggle datasets
    └── Case-Name/
```

## 1-Base-Algorithms

| # | Algorithm | Key concept | Dataset | Notebook |
|---|---|---|---|---|
| 01 | Linear Regression via Gradient Descent | Gradient descent · chain rule | Salary x Experience (course) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Base-Algorithms/01-Linear-Regression-Gradient-Descent/linear_regression_gd.ipynb) |
| 02 | MLP with Manual Backpropagation | Hidden layer · tanh · backprop | Noisy sine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Base-Algorithms/02-MLP-Manual-Backpropagation/mlp_manual_backprop.ipynb) |
| 03 | MLP with scikit-learn | Equivalent to R's `neuralnet` package | Noisy sine / generic function | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Base-Algorithms/03-MLP-Sklearn/mlp_sklearn.ipynb) |
| 04 | Polynomial Regression with Ridge | Overfitting · L2 penalty | Noisy sine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Base-Algorithms/04-Polynomial-Regression-Ridge/ridge_regularization.ipynb) |
| 05 | MLP Hyperparameter Search | Activation x neurons · bias-variance | Generic function | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Base-Algorithms/05-MLP-Hyperparameter-Search/mlp_hyperparameter_search.ipynb) |
| 🔜 | CNN (MNIST, Dogs vs Cats) | Convolutions · softmax · cross-entropy | MNIST / Dogs vs Cats | — |
| 🔜 | Financial time series | RNN · LSTM · GRU | Financial data | — |
| 🔜 | Text generation and classification | LSTM · NLP | IMDB | — |
| 🔜 | K-fold and Cross-Validation | Model evaluation | Boston Housing / Adult | — |
| 🔜 | Bias-Variance of the Estimator | Statistical theory | — | — |

## 2-Real-World-Kaggle-Cases

Not started yet — will apply the algorithms above to real Kaggle datasets, focused on problems close to my research area (social robotics / human-robot interaction).

## Running locally

```bash
git clone https://github.com/wdasilvamf/machine-learning-UFMG.git
cd machine-learning-UFMG
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab
```

Each algorithm folder is self-contained (`.py` + `.ipynb` + `data/`).

## Running on Google Colab

Click the "Open in Colab" badge for any algorithm in the table above. Each notebook's first cell detects the Colab environment and automatically clones the repository to access the `.py` module and the `data/` folder's dataset.

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

## Credits

*Original R scripts: Prof. Marcelo A. Costa — Deep Learning, UFMG*
*Python conversion and portfolio: Wanderson Filho — MSc candidate, Unifesp (Technological Innovation)*

The original R course materials are cited above for attribution but are **not redistributed** in this repository — they are course materials, not licensed for public redistribution. Everything in this repository (Python code, notebooks, documentation) is an original implementation of the same standard, well-known algorithms (gradient descent, backpropagation, Ridge regression, etc.), not a derivative of the R source code itself.
