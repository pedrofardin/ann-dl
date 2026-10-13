"""Configuração compartilhada pelos exercícios: semente, pastas e figuras."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera os PNG sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SEED = 42
CODE_DIR = Path(__file__).resolve().parent
FIGURES = CODE_DIR.parent / "figures"

CLASS_COLORS = ["tab:blue", "tab:orange", "tab:green"]


def save(fig, name):
    """Salva a figura em figures/ e libera a memória."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_points(ax, X, y, **style):
    """Desenha os pontos, uma cor por classe."""
    for c in np.unique(y):
        ax.scatter(*X[y == c].T, s=10, alpha=0.6, color=CLASS_COLORS[c], label=f"Classe {c}", **style)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def loss_figure(curves, title, name):
    """Perda de treino × época. curves: {rótulo da curva: lista com a perda de cada época}."""
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for label, hist in curves.items():
        ax.plot(np.arange(1, len(hist) + 1), hist, lw=2, label=label)
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de treino (entropia cruzada)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, name)
