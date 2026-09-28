"""Exercise 2 — Dados sobrepostos: o caso que o perceptron não resolve.

Treina o mesmo perceptron do Exercise 1 (mesma função train, eta = 0.01, 100 épocas), agora com
o rastreio do pocket ligado. Produz as Figuras 4, 5 e 6. Depois roda as verificações do item D:
onde a fronteira final fica, mais épocas, eta menor e a ordem das amostras.
"""

import matplotlib.pyplot as plt
import numpy as np

from common import (draw_boundary, generate, limits, plot_predictions, save, scatter_figure,
                    shade_regions)
from perceptron import accuracy, init_weights, predict, train

# Parâmetros do enunciado.
MEANS = [[3.0, 3.0], [4.0, 4.0]]
COV = [[1.5, 0.0], [0.0, 1.5]]
ETA = 0.01
MAX_EPOCHS = 100

# Reta de referência: a mediatriz entre as médias, x1 + x2 = 7. Com covariâncias iguais e
# isotrópicas, é a melhor reta para as distribuições do enunciado.
W_REF, B_REF = np.array([1.0, 1.0]), -7.0
CENTER = np.mean(MEANS, axis=0)  # centro da nuvem inteira: (3.5, 3.5)


def position(w, b, X):
    """Onde a reta w·x + b = 0 fica em relação à nuvem de pontos."""
    norm = np.linalg.norm(w)
    return {
        "offset": float(-b / norm),                    # distância da reta até a origem (com sinal)
        "center_distance": float((w @ CENTER + b) / norm),  # > 0: centro da nuvem no lado "classe 1"
        "predicted_1": float(predict(X, w, b).mean()),  # fração de pontos que o modelo chama de 1
        "b_over_norm_w": float(b / norm),
    }


def figure5(X, y, final, pocket):
    """Figura 5: as fronteiras final e pocket, um painel para os erros de cada uma."""
    xlim, ylim = limits(X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True, layout="constrained")
    panels = [(axes[0], final, "Pesos finais"), (axes[1], pocket, "Pesos do pocket")]
    for ax, (w, b, acc), title in panels:
        shade_regions(ax, w, b, xlim, ylim)
        plot_predictions(ax, X, y, w, b)
        draw_boundary(ax, final[0], final[1], color="crimson", lw=2.0, label="Fronteira final")
        draw_boundary(ax, pocket[0], pocket[1], color="black", lw=2.0, ls="--", label="Fronteira do pocket")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: acurácia = {100 * acc:.2f}%")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.suptitle("Figura 5 — Fronteiras de decisão final e do pocket")
    save(fig, "fig5_boundaries.png")


def figure6(run):
    """Figura 6: acurácia dos pesos atuais e melhor acurácia até agora (pocket), por época."""
    hist = run["history"]
    epochs = np.arange(1, run["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.plot(epochs, [100 * a for a in hist["accuracy"]], color="crimson", lw=1.4,
            label="Pesos atuais (fim de cada época)")
    ax.plot(epochs, [100 * a for a in hist["pocket_accuracy"]], color="black", lw=2, ls="--",
            label="Melhor até agora (pocket)")
    ax.axhline(50, color="gray", lw=0.8, ls=":", label="50% (chute)")
    ax.set_ylim(40, 80)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.legend(loc="center right")
    ax.set_title(f"Figura 6 — Acurácia × época ($\\eta$ = {ETA}, dados sobrepostos)")
    save(fig, "fig6_accuracy.png")


def run(rng):
    # A — gera os dados e desenha a Figura 4.
    X, y = generate(rng, MEANS, COV)
    scatter_figure(X, y, "Figura 4 — Dados sobrepostos: 1000 pontos por classe", "fig4_data.png")

    # B — a mesma função do Exercise 1; pocket=True liga a cópia do melhor (w, b).
    w0, b0 = init_weights(rng)
    main = train(X, y, w0, b0, ETA, max_epochs=MAX_EPOCHS, pocket=True)
    best = main["pocket"]

    # C — figuras.
    figure5(X, y, (main["w"], main["b"], main["accuracy"]), (best["w"], best["b"], best["accuracy"]))
    figure6(main)

    # D — o que acontece dentro da última época: treina 99 épocas e depois só a metade da
    # classe 0 da época 100. A ordem das amostras é a mesma, então as atualizações são as mesmas.
    first_99 = train(X, y, w0, b0, ETA, max_epochs=MAX_EPOCHS - 1)
    half = train(X[y == 0], y[y == 0], first_99["w"], first_99["b"], ETA, max_epochs=1)

    # D — mais épocas e eta menor, sem pocket (só interessa o que o laço deixa na mão).
    long = train(X, y, w0, b0, ETA, max_epochs=5 * MAX_EPOCHS)
    small_eta = train(X, y, w0, b0, ETA / 10, max_epochs=MAX_EPOCHS)

    # D — a mesma execução com a ordem das amostras embaralhada uma vez. É o último sorteio do
    # relatório, para não mudar nenhum número acima.
    order = rng.permutation(len(y))
    shuffled = train(X[order], y[order], w0, b0, ETA, max_epochs=MAX_EPOCHS, pocket=True)

    print("Exercise 2")
    print(f"  final: w = {main['w']}, b = {main['b']:.4f}, acurácia = {100 * main['accuracy']:.2f}%")
    print(f"  pocket: w = {best['w']}, b = {best['b']:.4f}, acurácia = {100 * best['accuracy']:.2f}% "
          f"(época {best['epoch']})")

    tail = np.array(long["history"]["accuracy"][-MAX_EPOCHS:])
    return {
        "w0": w0.tolist(),
        "b0": b0,
        "final": {"w": main["w"].tolist(), "b": main["b"], "accuracy": main["accuracy"],
                  **position(main["w"], main["b"], X)},
        "pocket": {"w": best["w"].tolist(), "b": best["b"], "accuracy": best["accuracy"],
                   "epoch": best["epoch"], "update": best["update"],
                   **position(best["w"], best["b"], X)},
        "reference_line": {"w": W_REF.tolist(), "b": B_REF, "accuracy": accuracy(X, y, W_REF, B_REF),
                           **position(W_REF, B_REF, X)},
        "epochs": main["epochs"],
        "converged": main["converged"],
        "total_updates": main["total_updates"],
        "updates_per_epoch": main["history"]["updates"],
        "min_updates_per_epoch": min(main["history"]["updates"]),
        "accuracy_per_epoch": main["history"]["accuracy"],
        "accuracy_range": [min(main["history"]["accuracy"]), max(main["history"]["accuracy"])],
        "mean_norm_x": float(np.linalg.norm(X, axis=1).mean()),
        "mid_last_epoch": {"accuracy": accuracy(X, y, half["w"], half["b"]),
                           **position(half["w"], half["b"], X)},
        "more_epochs": {"epochs": long["epochs"], "accuracy": long["accuracy"],
                        "last_100_range": [float(tail.min()), float(tail.max())],
                        "updates_last_epoch": long["history"]["updates"][-1],
                        **position(long["w"], long["b"], X)},
        "smaller_eta": {"eta": ETA / 10, "accuracy": small_eta["accuracy"],
                        "accuracy_range": [min(small_eta["history"]["accuracy"]),
                                           max(small_eta["history"]["accuracy"])],
                        **position(small_eta["w"], small_eta["b"], X)},
        "shuffled_order": {"accuracy": shuffled["accuracy"],
                           "pocket_accuracy": shuffled["pocket"]["accuracy"],
                           "accuracy_range": [min(shuffled["history"]["accuracy"]),
                                              max(shuffled["history"]["accuracy"])],
                           **position(shuffled["w"], shuffled["b"], X)},
    }
