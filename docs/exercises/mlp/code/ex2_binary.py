"""Exercício 2 — Classificação binária que uma reta não resolve.

Gera os dados, ajusta a linha de base linear (regressão logística), treina o MLP do zero e
produz as Figuras 1, 2 e 3.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import CLASS_COLORS, loss_figure, plot_points, save
from mlp import MLP

# Parâmetros do enunciado, sem alterações.
CENTERS = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
N_SAMPLES = [500, 250, 250]

# Hiperparâmetros escolhidos para o MLP.
LAYERS = [2, 8, 1]  # 2 entradas, 1 camada oculta de 8 neurônios tanh, 1 saída sigmoide
ETA = 0.5
EPOCHS = 2000


def generate():
    """A classe 0 é o grupo do centro; a classe 1 são os dois grupos das pontas."""
    X, c = make_blobs(n_samples=N_SAMPLES, centers=CENTERS, cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)  # classe 0: 1 grupo | classe 1: 2 grupos
    return X, y


def figure1(X, y):
    """Figura 1: os 1000 pontos coloridos por y."""
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot_points(ax, X, y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Figura 1 — Dados do Exercício 2: 1000 pontos, coloridos pela classe")
    ax.legend(loc="upper left")
    save(fig, "fig1_data.png")


def figure3(net, X_all, X_test, y_test, acc):
    """Figura 3: a fronteira de decisão do MLP, avaliada numa grade, sobre os pontos de teste."""
    pad = 1.0
    xx, yy = np.meshgrid(np.linspace(X_all[:, 0].min() - pad, X_all[:, 0].max() + pad, 400),
                         np.linspace(X_all[:, 1].min() - pad, X_all[:, 1].max() + pad, 400))
    p = net.predict_proba(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.contourf(xx, yy, p, levels=[0.0, 0.5, 1.0], colors=CLASS_COLORS[:2], alpha=0.18)
    ax.contour(xx, yy, p, levels=[0.5], colors="black", linewidths=1.5)
    plot_points(ax, X_test, y_test, edgecolor="white", linewidth=0.3)
    ax.plot([], [], color="black", lw=1.5, label="Fronteira $P(y=1) = 0.5$")
    ax.set_aspect("equal")
    ax.set_title(f"Figura 3 — Fronteira de decisão do MLP {LAYERS} (teste: {100 * acc:.1f}%)")
    ax.legend(loc="upper left")
    save(fig, "fig3_boundary.png")


def run(rng):
    # A — dados, divisão 80/20 estratificada e Figura 1.
    X, y = generate()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    figure1(X, y)

    # B — linha de base: uma única fronteira linear.
    baseline = LogisticRegression().fit(X_train, y_train)
    baseline_acc = float(baseline.score(X_test, y_test))
    p = baseline.predict_proba(X_test)[:, 1]

    # C — o MLP do zero. A inicialização dos pesos é o primeiro sorteio do rng do relatório.
    net = MLP(LAYERS, rng, hidden="tanh", output="sigmoid")
    history = net.fit(X_train, y_train, ETA, EPOCHS)
    train_acc, test_acc = net.accuracy(X_train, y_train), net.accuracy(X_test, y_test)

    loss_figure({f"MLP {LAYERS}, $\\eta$ = {ETA}": history},
                "Figura 2 — Perda de treino × época (Exercício 2)", "fig2_loss.png")
    figure3(net, X, X_test, y_test, test_acc)

    print("Exercício 2")
    print(f"  linha de base linear: teste = {100 * baseline_acc:.2f}%")
    print(f"  MLP {LAYERS}, eta = {ETA}, {EPOCHS} épocas: perda final = {history[-1]:.4f}, "
          f"treino = {100 * train_acc:.2f}%, teste = {100 * test_acc:.2f}%")

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "baseline_test_accuracy": baseline_acc,
        # Os pesos da reta e a faixa de P(y=1) que ela dá no teste: o item D usa os dois.
        "baseline_coef": baseline.coef_[0].tolist(),
        "baseline_intercept": float(baseline.intercept_[0]),
        "baseline_test_proba_range": [float(p.min()), float(p.max())],
        "baseline_test_predicted_class1": int(baseline.predict(X_test).sum()),
        "mlp": {
            "layers": LAYERS,
            "hidden_activation": "tanh",
            "output": "sigmoid + entropia cruzada binária",
            "n_params": net.n_params(),
            "eta": ETA,
            "epochs": EPOCHS,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        },
    }
