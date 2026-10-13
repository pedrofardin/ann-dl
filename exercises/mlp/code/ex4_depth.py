"""Exercício 4 — Uma segunda camada oculta ajuda?

Mesmos dados e mesma divisão do Exercício 3, mesmo número de épocas, mesma taxa de aprendizado.
A única mudança é a profundidade: [4, 16, 3] passa a [4, 16, 16, 3].

A comparação roda as duas redes com N_SEEDS inicializações diferentes. Cada repetição usa um
gerador filho do rng do relatório (rng.spawn), então tudo continua vindo da semente 42.
Produz a Figura 6.
"""

import matplotlib.pyplot as plt
import numpy as np

import ex3_multiclass as ex3
from common import save
from mlp import MLP

LAYERS = [4, 16, 16, 3]  # 2 camadas ocultas de 16 neurônios tanh
N_SEEDS = 5

# Experimento extra do item C: o mesmo par de redes com um orçamento de treino maior.
# Muda só o orçamento (eta e épocas), igual para as duas redes.
BIG_ETA = 0.5
BIG_EPOCHS = 4000


def train_eval(layers, gen, data, eta, epochs):
    """Treina uma rede e devolve a curva de perda e as métricas."""
    X_train, X_test, y_train, y_test = data
    net = MLP(layers, gen, hidden="tanh", output="softmax")
    history = net.fit(X_train, y_train, eta, epochs)
    return history, {
        "final_loss": history[-1],
        "train_accuracy": net.accuracy(X_train, y_train),
        "test_accuracy": net.accuracy(X_test, y_test),
        "n_params": net.n_params(),
    }


def compare(rng, data, eta, epochs):
    """Treina as duas profundidades com N_SEEDS geradores filhos; devolve curvas e médias."""
    runs = {"shallow": [], "deep": []}
    curves = {"shallow": [], "deep": []}
    for gen in rng.spawn(N_SEEDS):
        # O mesmo gerador filho inicializa as duas redes de uma repetição, uma depois da outra.
        for key, layers in (("shallow", ex3.LAYERS), ("deep", LAYERS)):
            history, metrics = train_eval(layers, gen, data, eta, epochs)
            runs[key].append(metrics)
            curves[key].append(history)

    def stats(key, field):
        v = np.array([r[field] for r in runs[key]])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}

    summary = {
        key: {"layers": ex3.LAYERS if key == "shallow" else LAYERS,
              "n_params": runs[key][0]["n_params"],
              **{f: stats(key, f) for f in ("final_loss", "train_accuracy", "test_accuracy")}}
        for key in runs
    }
    diffs = np.array([d["test_accuracy"] - s["test_accuracy"] for s, d in zip(runs["shallow"], runs["deep"])])
    summary["test_accuracy_diff_deep_minus_shallow"] = {
        "mean": float(diffs.mean()), "std": float(diffs.std(ddof=1)), "values": diffs.tolist()}
    return summary, curves


def figure6(main_shallow, main_deep, curves):
    """Figura 6: perda de treino das duas redes nos mesmos eixos.

    Linhas grossas: a rodada principal de cada exercício. Linhas finas: as N_SEEDS repetições.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    epochs = np.arange(1, len(main_shallow) + 1)
    for key, main, color, label in (("shallow", main_shallow, "tab:blue", f"Exercício 3: {ex3.LAYERS}"),
                                    ("deep", main_deep, "tab:red", f"Exercício 4: {LAYERS}")):
        for i, h in enumerate(curves[key]):
            ax.plot(epochs, h, color=color, lw=0.8, alpha=0.3,
                    label=f"{label}, {N_SEEDS} sementes" if i == 0 else None)
        ax.plot(epochs, main, color=color, lw=2.2, label=f"{label}, rodada principal")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda de treino (entropia cruzada)")
    ax.set_ylim(0, None)
    ax.set_title(f"Figura 6 — Perda de treino: 1 × 2 camadas ocultas ($\\eta$ = {ex3.ETA})")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "fig6_depth_loss.png")


def run(rng, shallow_history):
    X, y = ex3.generate()
    data = ex3.split(X, y)

    # A — a rede mais profunda, com o mesmo orçamento de treino do Exercício 3.
    main_history, main = train_eval(LAYERS, rng, data, ex3.ETA, ex3.EPOCHS)

    # B — comparação controlada com N_SEEDS inicializações.
    summary, curves = compare(rng, data, ex3.ETA, ex3.EPOCHS)
    figure6(shallow_history, main_history, curves)

    # C — o mesmo par com um orçamento maior: a perda de treino e a acurácia contam a mesma história?
    big, _ = compare(rng, data, BIG_ETA, BIG_EPOCHS)

    print("Exercício 4")
    print(f"  MLP {LAYERS}, eta = {ex3.ETA}, {ex3.EPOCHS} épocas: perda final = {main['final_loss']:.4f}, "
          f"treino = {100 * main['train_accuracy']:.2f}%, teste = {100 * main['test_accuracy']:.2f}%")
    for name, s in (("orçamento do Ex. 3", summary), (f"eta = {BIG_ETA}, {BIG_EPOCHS} épocas", big)):
        print(f"  {name}, média de {N_SEEDS} sementes:")
        for key in ("shallow", "deep"):
            print(f"    {str(s[key]['layers']):14s} perda = {s[key]['final_loss']['mean']:.4f}, "
                  f"teste = {100 * s[key]['test_accuracy']['mean']:.2f}% ± {100 * s[key]['test_accuracy']['std']:.2f}")
        d = s["test_accuracy_diff_deep_minus_shallow"]
        print(f"    diferença (profunda - rasa) = {100 * d['mean']:+.2f} p.p. ± {100 * d['std']:.2f}")

    return {
        "main": {"layers": LAYERS, "eta": ex3.ETA, "epochs": ex3.EPOCHS, **main},
        "n_seeds": N_SEEDS,
        "comparison": summary,
        "bigger_budget": {"eta": BIG_ETA, "epochs": BIG_EPOCHS, **big},
    }
