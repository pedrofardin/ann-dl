"""Configuração compartilhada pelos dois exercícios: semente, pastas, dados e figuras."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera os PNG sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from perceptron import predict  # noqa: E402

SEED = 42
CODE_DIR = Path(__file__).resolve().parent
FIGURES = CODE_DIR.parent / "figures"

N_PER_CLASS = 1000
CLASS_COLORS = ["tab:blue", "tab:orange"]


def save(fig, name):
    """Salva a figura em figures/ e libera a memória."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate(rng, means, cov):
    """Amostra 1000 pontos por classe de N(means[k], cov).

    A ordem fica a da geração: 1000 pontos da classe 0 e depois 1000 da classe 1. O perceptron
    visita as amostras nessa ordem em toda época (o enunciado não pede embaralhamento).
    """
    # method="cholesky": a decomposição de Cholesky é única, então o resultado não muda de máquina.
    X = np.vstack([rng.multivariate_normal(m, cov, size=N_PER_CLASS, method="cholesky") for m in means])
    y = np.repeat([0, 1], N_PER_CLASS)
    return X, y


def plot_points(ax, X, y):
    """Desenha os pontos das duas classes, uma cor por classe."""
    for c in (0, 1):
        ax.scatter(*X[y == c].T, s=8, alpha=0.5, color=CLASS_COLORS[c], label=f"Classe {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def scatter_figure(X, y, title, name):
    """Figuras 1 e 4: dispersão dos 2000 pontos, uma cor por classe."""
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot_points(ax, X, y)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.legend(loc="upper left")
    save(fig, name)


def draw_boundary(ax, w, b, **style):
    """Desenha a reta w·x + b = 0 inteira, sem mudar os limites dos eixos.

    p = -b w / ||w||² é o ponto da reta mais perto da origem; (-w2, w1) é a direção da reta.
    """
    p = -b * w / (w @ w)
    return ax.axline(p, p + np.array([-w[1], w[0]]), **style)


def shade_regions(ax, w, b, xlim, ylim):
    """Pinta de leve a região que o modelo prevê como classe 0 e a que prevê como classe 1."""
    xx, yy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    regions = predict(np.column_stack([xx.ravel(), yy.ravel()]), w, b).reshape(xx.shape)
    ax.contourf(xx, yy, regions, levels=[-0.5, 0.5, 1.5], colors=CLASS_COLORS, alpha=0.08)


def plot_predictions(ax, X, y, w, b):
    """Desenha os pontos com a cor da classe verdadeira: bolinha se (w, b) acerta, x se erra."""
    wrong = predict(X, w, b) != y
    for c in (0, 1):
        ok, bad = (y == c) & ~wrong, (y == c) & wrong
        ax.scatter(*X[ok].T, s=8, alpha=0.5, color=CLASS_COLORS[c], label=f"Classe {c}, acerto")
        ax.scatter(*X[bad].T, marker="x", s=18, lw=0.9, color=CLASS_COLORS[c],
                   label=f"Classe {c}, erro ({bad.sum()})")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def limits(X, pad=0.8):
    """Limites dos eixos que cobrem todos os pontos, com uma margem."""
    return ((X[:, 0].min() - pad, X[:, 0].max() + pad),
            (X[:, 1].min() - pad, X[:, 1].max() + pad))
