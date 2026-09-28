"""Exercise 1 — Dados separáveis: o caso para o qual o perceptron foi projetado.

Gera as duas classes, treina o perceptron com eta = 0.01 e com eta = 1.0 (mesmo w inicial)
e confere numericamente o argumento do item D sobre a partida de w = 0.
Produz as Figuras 1, 2 e 3 e uma figura auxiliar do item D.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import (draw_boundary, generate, limits, plot_points, plot_predictions, save,
                    scatter_figure, shade_regions)
from perceptron import init_weights, train

# Parâmetros do enunciado.
MEANS = [[1.5, 1.5], [5.0, 5.0]]
COV = [[0.5, 0.0], [0.0, 0.5]]
ETA = 0.01
ETA_BIG = 1.0


def direction(w):
    """Vetor unitário w / ||w||: a direção normal à fronteira, sem a escala."""
    return w / np.linalg.norm(w)


def angle_deg(u, v):
    """Ângulo, em graus, entre as direções de u e v."""
    cos = np.clip(direction(u) @ direction(v), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def figure2(X, y, run):
    """Figura 2: a fronteira w·x + b = 0 sobre os pontos, com os mal classificados marcados."""
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    xlim, ylim = limits(X)
    shade_regions(ax, run["w"], run["b"], xlim, ylim)
    plot_predictions(ax, X, y, run["w"], run["b"])
    draw_boundary(ax, run["w"], run["b"], color="black", lw=1.8, label="Fronteira $w \\cdot x + b = 0$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(f"Figura 2 — Fronteira de decisão ($\\eta$ = {ETA}), acurácia = {100 * run['accuracy']:.2f}%")
    ax.legend(loc="upper left")
    save(fig, "fig2_boundary.png")


def figure3(run):
    """Figura 3: acurácia × época. As barras cinzas mostram as atualizações feitas em cada época."""
    hist = run["history"]
    epochs = np.arange(1, run["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")

    ax2 = ax.twinx()  # eixo da direita: número de atualizações
    ax2.bar(epochs, hist["updates"], color="gray", alpha=0.35, label="Atualizações na época")
    ax2.set_ylabel("Atualizações na época")
    ax2.set_ylim(0, max(hist["updates"]) * 2.5)  # barras baixas, para não cobrir a curva

    ax.plot(epochs, [100 * a for a in hist["accuracy"]], "o-", color="tab:purple", lw=2,
            label="Acurácia no dataset completo")
    ax.set_zorder(ax2.get_zorder() + 1)  # curva na frente das barras
    ax.patch.set_visible(False)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.set_xticks(epochs[::2] if len(epochs) > 15 else epochs)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.set_title(f"Figura 3 — Acurácia × época ($\\eta$ = {ETA})")
    save(fig, "fig3_accuracy.png")


def figure_eta(X, y, small, big):
    """Figura auxiliar do item D: as fronteiras de eta = 0.01 e eta = 1.0, com zoom no vão entre as nuvens."""
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    plot_points(ax, X, y)
    draw_boundary(ax, small["w"], small["b"], color="black", lw=1.8, label=f"$\\eta$ = {ETA}")
    draw_boundary(ax, big["w"], big["b"], color="crimson", lw=1.8, ls="--", label=f"$\\eta$ = {ETA_BIG}")
    ax.set_xlim(1.0, 5.5)
    ax.set_ylim(1.0, 5.5)
    ax.set_aspect("equal")
    ax.set_title("Figura auxiliar (item D) — Fronteiras com $\\eta$ = 0.01 e $\\eta$ = 1.0 (zoom)")
    ax.legend(loc="upper left")
    save(fig, "fig_d_eta.png")


def summary(run):
    """Números de uma execução, prontos para o results.json."""
    return {
        "w": run["w"].tolist(),
        "b": run["b"],
        "direction": direction(run["w"]).tolist(),
        "offset": float(-run["b"] / np.linalg.norm(run["w"])),  # distância da reta até a origem
        "epochs": run["epochs"],
        "converged": run["converged"],
        "total_updates": run["total_updates"],
        "accuracy": run["accuracy"],
        "updates_per_epoch": run["history"]["updates"],
        "accuracy_per_epoch": run["history"]["accuracy"],
    }


def run(rng):
    # A — gera os dados e desenha a Figura 1.
    X, y = generate(rng, MEANS, COV)
    scatter_figure(X, y, "Figura 1 — Dados separáveis: 1000 pontos por classe", "fig1_data.png")

    # B/C — um único sorteio de w inicial, reutilizado pelas duas taxas de aprendizado.
    w0, b0 = init_weights(rng)
    small = train(X, y, w0, b0, ETA)
    figure2(X, y, small)
    figure3(small)

    # D — mesma execução, só com eta = 1.0: mesmos dados, mesma ordem, mesmo w0.
    big = train(X, y, w0, b0, ETA_BIG)
    figure_eta(X, y, small, big)

    # D — partida do zero: o argumento algébrico diz que w e b escalam com eta2 / eta1.
    zero_small = train(X, y, np.zeros(2), 0.0, ETA)
    zero_big = train(X, y, np.zeros(2), 0.0, ETA_BIG)
    ratio = ETA_BIG / ETA

    print("Exercise 1")
    for name, r in [("eta = 0.01", small), ("eta = 1.0", big)]:
        print(f"  {name}: w = {r['w']}, b = {r['b']:.4f}, épocas = {r['epochs']}, "
              f"acurácia = {100 * r['accuracy']:.2f}%")

    return {
        "w0": w0.tolist(),
        "b0": b0,
        "eta_small": summary(small),
        "eta_big": summary(big),
        "angle_between_directions_deg": angle_deg(small["w"], big["w"]),
        "zero_start": {
            "eta_small": summary(zero_small),
            "eta_big": summary(zero_big),
            "expected_ratio": ratio,
            # Quanto os pesos de eta = 1.0 diferem de (eta2/eta1) × pesos de eta = 0.01.
            "max_abs_diff_w": float(np.abs(zero_big["w"] - ratio * zero_small["w"]).max()),
            "abs_diff_b": abs(zero_big["b"] - ratio * zero_small["b"]),
            "same_epochs": zero_small["epochs"] == zero_big["epochs"],
            "angle_deg": angle_deg(zero_small["w"], zero_big["w"]),
        },
    }
