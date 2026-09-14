"""Configuração compartilhada pelos três exercícios: semente, pastas e gravação das figuras."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera os PNG sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
CODE_DIR = Path(__file__).resolve().parent
FIGURES = CODE_DIR.parent / "figures"
DATA_DIR = CODE_DIR / "data"


def save(fig, name):
    """Salva a figura em figures/ e libera a memória."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
