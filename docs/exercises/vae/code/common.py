"""Configuração compartilhada pelos exercícios: semente, pastas, dados e figuras."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera os PNG sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402

SEED = 42
CODE_DIR = Path(__file__).resolve().parent
FIGURES = CODE_DIR.parent / "figures"
DATA_DIR = CODE_DIR.parents[3] / "data"  # raiz do repositório; fica fora do Git (.gitignore)

torch.use_deterministic_algorithms(True)  # mesma execução, mesmos números


def _stack(subset):
    """Junta as imagens de um Dataset num tensor (N, 784) e os rótulos num tensor (N,)."""
    xs, ys = next(iter(DataLoader(subset, batch_size=len(subset))))
    return xs.view(len(subset), -1), ys


def load_mnist():
    """Item 1A: MNIST com pixels em [0, 1], achatado em vetores de 784.

    A divisão 50 000 / 10 000 é exatamente a do enunciado. O conjunto de teste volta separado e
    só é usado depois do treino, para os números finais e as figuras.
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))
    return {"train": _stack(train), "val": _stack(val), "test": _stack(test)}


def save(fig, name):
    """Salva a figura em figures/ e libera a memória."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tile(images, rows, cols):
    """Monta (rows·cols, 784) imagens num mosaico (rows·28, cols·28), na ordem das linhas."""
    imgs = np.asarray(images).reshape(rows, cols, 28, 28)
    return imgs.transpose(0, 2, 1, 3).reshape(rows * 28, cols * 28)


def loss_curves_figure(history, title, name):
    """Dois painéis: reconstrução × época e KL × época, cada um com treino e validação."""
    epochs = np.arange(1, len(history["train_rec"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, term, label in [(axes[0], "rec", "Reconstrução (BCE)"), (axes[1], "kl", "KL")]:
        ax.plot(epochs, history[f"train_{term}"], "o-", lw=2, ms=4, label="Treino (média da época)")
        ax.plot(epochs, history[f"val_{term}"], "s-", lw=2, ms=4, label="Validação (fim da época)")
        ax.set_xlabel("Época")
        ax.set_ylabel(f"{label} (nats por imagem)")
        ax.set_title(f"{label} × época")
        ax.set_xticks(epochs[::2])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    save(fig, name)
