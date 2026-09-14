"""Exercise 1 — Point clouds: geometry and spread in 2D.

Gera as 4 classes gaussianas do enunciado, repete a geração com os desvios padrão
multiplicados por s e mede o quanto as nuvens se misturam. Produz as Figuras 1, 2 e 3.
Nenhum modelo é treinado: todas as medidas são geométricas.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from common import save

# Parâmetros do enunciado: média (x, y) e desvio padrão (x, y) de cada classe.
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
CLASSES = range(len(MEANS))
SCALES = [0.5, 1.0, 2.0, 4.0]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def generate(rng, s=1.0):
    """Amostra 100 pontos por classe. As médias ficam fixas; só o desvio muda (× s)."""
    X = np.vstack([rng.normal(MEANS[k], STDS[k] * s, size=(N_PER_CLASS, 2)) for k in CLASSES])
    y = np.repeat(np.arange(len(MEANS)), N_PER_CLASS)
    return X, y


def separation_ratios(s=1.0):
    """r_ij = ||mu_i - mu_j|| / (sigma_i + sigma_j), com sigma_k = média dos dois desvios da classe k."""
    sigma_bar = (STDS * s).mean(axis=1)
    return {
        (i, j): float(np.linalg.norm(MEANS[i] - MEANS[j]) / (sigma_bar[i] + sigma_bar[j]))
        for i, j in combinations(CLASSES, 2)
    }


def nearest_center(X):
    """Índice do centro de classe (média do enunciado) mais próximo de cada ponto."""
    # dist[n, k]: distância euclidiana do ponto n ao centro da classe k.
    dist = np.linalg.norm(X[:, None, :] - MEANS[None, :, :], axis=2)
    return dist.argmin(axis=1)


def mixing_rate(X, y):
    """Fração de pontos cujo centro mais próximo não é o da própria classe."""
    return float(np.mean(nearest_center(X) != y))


def mixing_counts(X, y):
    """counts[k, m]: pontos da classe k cujo centro mais próximo é o da classe m."""
    counts = np.zeros((len(MEANS), len(MEANS)), dtype=int)
    np.add.at(counts, (y, nearest_center(X)), 1)
    return counts


def most_likely_class(xx, yy):
    """Classe de maior densidade em cada ponto da grade, com as gaussianas verdadeiras (s = 1).

    Não é um modelo treinado: usa só os parâmetros do enunciado. Serve de esboço da
    fronteira que uma rede bem treinada tende a aproximar.
    """
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    z = (grid[:, None, :] - MEANS[None]) / STDS[None]
    # log da densidade gaussiana com covariância diagonal, sem a constante comum a todas as classes.
    log_density = -0.5 * (z**2).sum(axis=2) - np.log(STDS).sum(axis=1)
    return log_density.argmax(axis=1).reshape(xx.shape)


def plot_clouds(ax, X, y, marker_size=14):
    """Desenha os pontos de cada classe e marca o centro (média) de cada nuvem."""
    for k in CLASSES:
        ax.scatter(*X[y == k].T, s=marker_size, alpha=0.65, color=COLORS[k], label=f"Classe {k}")
    ax.scatter(*MEANS.T, marker="X", s=140, color="black", edgecolors="white",
               linewidths=1.0, zorder=3, label="Centro (média)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")


def figure1(X, y, name, title, boundaries=False):
    """Figura 1: as 4 nuvens com os centros. Com boundaries=True, acrescenta o esboço do item C."""
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    xlim, ylim = (-2.0, 18.0), (-6.0, 12.0)
    handles = []
    if boundaries:
        xx, yy = np.meshgrid(np.linspace(*xlim, 800), np.linspace(*ylim, 800))
        regions = most_likely_class(xx, yy)
        ax.contourf(xx, yy, regions, levels=np.arange(-0.5, len(MEANS)),
                    cmap=ListedColormap(COLORS), alpha=0.12)
        for k in CLASSES:  # contorno de cada região: as linhas se sobrepõem na fronteira
            ax.contour(xx, yy, (regions == k).astype(float), levels=[0.5], colors="black", linewidths=1.4)
        handles.append(Line2D([], [], color="black", lw=1.4, label="Fronteira esboçada"))
    plot_clouds(ax, X, y)
    for k in CLASSES:
        ax.annotate(f"$\\mu_{k}$", MEANS[k], textcoords="offset points", xytext=(8, 8),
                    fontsize=12, fontweight="bold")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    point_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=point_handles + handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, name)


def figure2(datasets, rates):
    """Figura 2: um painel por fator de escala, todos com os mesmos limites de eixo."""
    points = np.vstack([X for X, _ in datasets.values()])
    pad = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True, layout="constrained")
    for ax, s in zip(axes.flat, SCALES):
        plot_clouds(ax, *datasets[s], marker_size=8)
        ax.set_title(f"s = {s}  —  mixing rate = {100 * rates[s]:.2f}%")
        ax.set_xlim(points[:, 0].min() - pad, points[:, 0].max() + pad)
        ax.set_ylim(points[:, 1].min() - pad, points[:, 1].max() + pad)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle("Figura 2 — As mesmas 4 classes em 4 fatores de escala (eixos compartilhados)")
    save(fig, "fig2_scales.png")


def figure3(rates, r_min, pair):
    """Figura 3: mixing rate × s, com o menor r_ij (que cai como 1/s) no eixo da direita."""
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    values = [100 * rates[s] for s in SCALES]
    ax.plot(SCALES, values, "o-", color="tab:purple", lw=2, label="Mixing rate (%)")
    for s, v in zip(SCALES, values):
        ax.annotate(f"{v:.2f}%", (s, v), textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set_ylim(-3, max(values) + 8)  # espaço para os rótulos acima dos pontos
    ax.set_xscale("log", base=2)
    ax.set_xticks(SCALES, [str(s) for s in SCALES])
    ax.set_xlabel("Fator de escala s (escala log)")
    ax.set_ylabel("Mixing rate (%)")

    ax2 = ax.twinx()
    ax2.plot(SCALES, [r_min / s for s in SCALES], "s--", color="gray",
             label=f"Menor $r_{{ij}}$ (par {pair[0]}–{pair[1]}) = {r_min:.2f} / s")
    ax2.axhline(1.0, color="gray", lw=0.8, ls=":", label="$r_{ij} = 1$")
    ax2.set_ylabel("Menor $r_{ij}$")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left")
    ax.set_title("Figura 3 — Mixing rate × fator de escala s")
    save(fig, "fig3_mixing_rate.png")


def run(rng):
    # A — gera as nuvens originais (s = 1) e desenha a Figura 1.
    X, y = generate(rng, 1.0)
    figure1(X, y, "fig1_point_clouds.png", "Figura 1 — Nuvens de pontos das 4 classes (s = 1)")

    # B — as mesmas 4 classes em 4 escalas. O dataset de s = 1 é o próprio dataset do item A.
    datasets = {s: (X, y) if s == 1.0 else generate(rng, s) for s in SCALES}
    rates = {s: mixing_rate(*datasets[s]) for s in SCALES}
    ratios = separation_ratios(1.0)
    pair, r_min = min(ratios.items(), key=lambda item: item[1])
    figure2(datasets, rates)
    figure3(rates, r_min, pair)

    # C — esboço das fronteiras sobre a Figura 1.
    figure1(X, y, "fig1_boundaries.png",
            "Figura 1 (item C) — Esboço das fronteiras de decisão sobre as nuvens (s = 1)",
            boundaries=True)

    print("Exercise 1")
    for (i, j), r in ratios.items():
        print(f"  r_{i}{j} = {r:.3f}")
    for s in SCALES:
        print(f"  s = {s}: mixing rate = {100 * rates[s]:.2f}%")

    return {
        "separation_ratios_s1": {f"{i}-{j}": r for (i, j), r in ratios.items()},
        "smallest_ratio_pair": f"{pair[0]}-{pair[1]}",
        "smallest_ratio_s1": r_min,
        "smallest_ratio_s2": r_min / 2,
        "smallest_ratio_s2_direct": separation_ratios(2.0)[pair],  # confere que r escala com 1/s
        "mixing_rate": {str(s): rates[s] for s in SCALES},
        "mixing_counts": {str(s): mixing_counts(*datasets[s]).tolist() for s in SCALES},
    }
