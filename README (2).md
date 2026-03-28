# Machine Learning — UFMG
### Deep Learning Course | Prof. Marcelo A. Costa
**Python conversion & portfolio** · [wdasilvamf](https://github.com/wdasilvamf)

---

This repository contains Python conversions of the R scripts from the Deep Learning course at UFMG. Each script is implemented **from scratch** — no high-level abstractions — to build a solid understanding of the underlying mathematics before moving to frameworks.

The series follows a deliberate progression: from linear models to MLPs, regularization, Keras, and eventually computer vision applied to social robotics research.

---

## Scripts

| # | Topic | Key Concept | Dataset | Notebook |
|---|-------|-------------|---------|----------|
| 01 | Linear Regression | Gradient descent · Chain rule | Housing Prices (Kaggle) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/script01/script01_backpropagation_linear.ipynb) |
| 02 | MLP from Scratch | Hidden layers · tanh · Backprop | Sine function (noisy) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/script02/script02_mlp_sine_backpropagation.ipynb) |
| 03 | Ridge Regularization | Overfitting · λ penalty | Sine function (noisy) | 🔜 |
| 04 | MLP with Keras | Framework abstraction | Sine function (noisy) | 🔜 |
| 05 | MNIST Classification | CNNs · Softmax · Cross-entropy | MNIST | 🔜 |

---

## Learning Progression

```
Script 01              Script 02/03/04         Script 05+
Linear model     →     MLP from scratch    →   CNNs / MNIST
(no hidden layer)      (tanh, backprop,        (applied to
                        regularization,         gaze analysis —
                        Keras)                  Furhat robot)
```

---

## Repository Structure

```
machine-learning-UFMG/
│
├── script01/
│   ├── script01_backpropagation_linear.ipynb
│   ├── script01_backpropagation_linear.py
│   └── Housing.csv
│
├── script02/
│   ├── script02_mlp_sine_backpropagation.ipynb
│   ├── script02_mlp_sine_backpropagation.py
│   └── Seno.xlsx
│
├── script03/                    # coming soon
├── script04/                    # coming soon
├── script05/                    # coming soon
│
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/wdasilvamf/machine-learning-UFMG.git
cd machine-learning-UFMG
pip install numpy pandas matplotlib scikit-learn openpyxl
```

Each script folder is self-contained. Place the dataset file in the same folder before running.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

---

*Original R scripts: Prof. Marcelo A. Costa — Deep Learning, UFMG*  
*Python conversion: Wanderson Filho — MSc candidate, Unifesp (Technological Innovation)*
