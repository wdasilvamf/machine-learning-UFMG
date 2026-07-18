# Machine Learning — UFMG
### Deep Learning Course | Prof. Marcelo A. Costa
**Tradução R → Python & Portfólio** · [wdasilvamf](https://github.com/wdasilvamf)

---

Conversão em Python dos scripts em R do curso de Deep Learning da UFMG, com implementações **from scratch** (sem abstrações de alto nível) para consolidar o entendimento da matemática por trás de cada modelo antes de partir para frameworks.

O portfólio segue uma progressão deliberada: de modelos lineares até MLPs, regularização, Keras e, eventualmente, visão computacional aplicada à minha pesquisa de mestrado em Inovação Tecnológica (Unifesp), com foco em sistemas cognitivos e robótica social.

## Estrutura

```
├── 1-Algoritmos-Base/          # Traducao fiel dos scripts do curso, um algoritmo por pasta
│   └── NN-Nome-Algoritmo/
│       ├── algoritmo.py        # Implementacao modular (funcoes + docstrings)
│       ├── algoritmo.ipynb     # Teoria, hiperparametros e interpretacao de metricas
│       └── data/                # Dataset original do script em R
│
└── 2-Casos-Reais-Kaggle/       # Os mesmos algoritmos aplicados a datasets reais do Kaggle
    └── Nome-do-Caso/
```

## 1-Algoritmos-Base

| # | Algoritmo | Conceito-chave | Dataset | Notebook |
|---|---|---|---|---|
| 01 | Regressão Linear via Gradient Descent | Gradient descent · regra da cadeia | Salário x Experiência (curso) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Algoritmos-Base/01-Regressao-Linear-Gradient-Descent/regressao_linear_gd.ipynb) |
| 02 | MLP com Backpropagation Manual | Camada escondida · tanh · backprop | Seno ruidoso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Algoritmos-Base/02-MLP-Backpropagation-Manual/mlp_backprop_manual.ipynb) |
| 03 | MLP com scikit-learn | Equivalente ao pacote `neuralnet` do R | Seno ruidoso / função genérica | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Algoritmos-Base/03-MLP-Neuralnet-Sklearn/mlp_sklearn.ipynb) |
| 04 | Regressão Polinomial com Ridge | Overfitting · penalidade L2 | Seno ruidoso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Algoritmos-Base/04-Regressao-Polinomial-Ridge/ridge_regularizacao.ipynb) |
| 05 | Busca de Hiperparâmetros para MLP | Ativação x neurônios · viés-variância | Função genérica | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wdasilvamf/machine-learning-UFMG/blob/main/1-Algoritmos-Base/05-MLP-Hyperparameter-Search/mlp_hyperparameter_search.ipynb) |
| 🔜 | CNN (MNIST, Dogs vs Cats) | Convoluções · softmax · cross-entropy | MNIST / Dogs vs Cats | — |
| 🔜 | Séries temporais financeiras | RNN · LSTM · GRU | dados financeiros | — |
| 🔜 | Geração de texto e classificação | LSTM · NLP | IMDB | — |
| 🔜 | K-fold e Validação Cruzada | Avaliação de modelos | Boston Housing / Adult | — |
| 🔜 | Viés-Variância do Estimador | Teoria estatística | — | — |

## 2-Casos-Reais-Kaggle

Ainda não iniciado — vai aplicar os algoritmos acima a datasets reais do Kaggle, com foco em problemas próximos da minha área de pesquisa (robótica social / interação humano-robô).

## Rodando localmente

```bash
git clone https://github.com/wdasilvamf/machine-learning-UFMG.git
cd machine-learning-UFMG
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab
```

Cada pasta de algoritmo é autocontida (`.py` + `.ipynb` + `data/`).

## Rodando no Google Colab

Clique no badge "Open in Colab" de cada algoritmo na tabela acima. A primeira célula de cada notebook detecta o ambiente Colab e clona o repositório automaticamente para ter acesso ao módulo `.py` e ao dataset da pasta `data/`.

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

## Créditos

*Scripts originais em R: Prof. Marcelo A. Costa — Deep Learning, UFMG*
*Conversão para Python e portfólio: Wanderson Filho — Mestrando em Inovação Tecnológica, Unifesp*
