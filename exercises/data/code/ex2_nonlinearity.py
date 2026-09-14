"""Exercise 2 — Non-linearity in higher dimensions.

Dataset I: duas gaussianas 5D deslocadas. Dataset II: núcleo e casca concêntricos em 5D.
Compara os dois com PCA (Figura 4), distância entre centros e histograma do raio (Figura 5).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from common import save

N_PER_CLASS = 500
DIM = 5

MU_A = np.zeros(DIM)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(DIM, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# Raio das cascas: rho ~ N(média, 0.4), com 0.4 lido como desvio padrão (convenção do NumPy).
RADIUS_C, RADIUS_D, RADIUS_STD = 2.0, 5.0, 0.4
THRESHOLD = (RADIUS_C + RADIUS_D) / 2  # raio no meio do caminho entre núcleo e casca: 3.5

CLASS_COLORS = ["tab:blue", "tab:red"]


def check_covariance(cov):
    """Uma covariância válida é simétrica e positiva definida (todos os autovalores > 0)."""
    eigenvalues = np.linalg.eigvalsh(cov)
    if not np.allclose(cov, cov.T) or eigenvalues.min() <= 0:
        raise ValueError(f"covariância inválida, autovalores = {eigenvalues}")
    return float(eigenvalues.min())


def dataset_gaussians(rng):
    """Dataset I: classe A ~ N(mu_A, Sigma_A) e classe B ~ N(mu_B, Sigma_B), 500 pontos cada."""
    # method="cholesky": a decomposição de Cholesky é única, então o resultado não muda de máquina.
    Xa = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS, method="cholesky")
    Xb = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS, method="cholesky")
    return np.vstack([Xa, Xb]), np.repeat([0, 1], N_PER_CLASS)


def shell(rng, radius_mean):
    """Pontos x = rho * u, com u uniforme na esfera unitária de R^5."""
    v = rng.standard_normal((N_PER_CLASS, DIM))
    u = v / np.linalg.norm(v, axis=1, keepdims=True)  # cada linha passa a ter norma 1
    rho = rng.normal(radius_mean, RADIUS_STD, size=N_PER_CLASS)
    return rho[:, None] * u


def dataset_shells(rng):
    """Dataset II: classe C (núcleo, raio ~2) e classe D (casca, raio ~5), 500 pontos cada."""
    return np.vstack([shell(rng, RADIUS_C), shell(rng, RADIUS_D)]), np.repeat([0, 1], N_PER_CLASS)


def radial_rule(X):
    """f(x) = ||x||^2 - 3.5^2. f > 0 prevê casca (D); f < 0 prevê núcleo (C)."""
    return (X**2).sum(axis=1) - THRESHOLD**2


def figure4(datasets):
    """Figura 4: PCA em 2D de cada dataset, lado a lado. Devolve a variância explicada."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    explained = {}
    for ax, (key, (title, X, y, names)) in zip(axes, datasets.items()):
        pca = PCA(n_components=2, svd_solver="full").fit(X)  # o PCA centraliza os dados sozinho
        Z = pca.transform(X)
        ratio = pca.explained_variance_ratio_
        explained[key] = ratio.tolist()
        for c in (0, 1):
            ax.scatter(*Z[y == c].T, s=10, alpha=0.55, color=CLASS_COLORS[c], label=names[c])
        ax.set_title(f"{title}: PC1 + PC2 = {100 * ratio.sum():.1f}% da variância")
        ax.set_xlabel(f"PC1 ({100 * ratio[0]:.1f}% da variância)")
        ax.set_ylabel(f"PC2 ({100 * ratio[1]:.1f}% da variância)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper right")
    fig.suptitle("Figura 4 — Projeção PCA em 2D dos dois datasets 5D")
    save(fig, "fig4_pca.png")
    return explained


def figure5(datasets, radii):
    """Figura 5: histograma do raio ||x|| em 5D, com as duas classes no mesmo eixo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, (key, (title, _, y, names)) in zip(axes, datasets.items()):
        bins = np.histogram_bin_edges(radii[key], bins=40)
        for c in (0, 1):
            ax.hist(radii[key][y == c], bins=bins, alpha=0.6, color=CLASS_COLORS[c], label=names[c])
        ax.set_title(title)
        ax.set_xlabel("Raio $\\|x\\|$ (calculado em 5D)")
        ax.set_ylabel("Número de pontos")
        ax.legend(loc="best")
    axes[1].axvline(THRESHOLD, color="black", ls="--", lw=1.2)
    axes[1].annotate(f"$\\|x\\| = {THRESHOLD}$", (THRESHOLD, axes[1].get_ylim()[1] * 0.9),
                     xytext=(6, 0), textcoords="offset points")
    fig.suptitle("Figura 5 — Histograma do raio de cada ponto, por classe")
    save(fig, "fig5_radius.png")


def run(rng):
    # A — Dataset I. Antes de amostrar, confere se as covariâncias do enunciado são válidas.
    min_eigenvalues = {"Sigma_A": check_covariance(SIGMA_A), "Sigma_B": check_covariance(SIGMA_B)}
    X1, y1 = dataset_gaussians(rng)

    # B — Dataset II.
    X2, y2 = dataset_shells(rng)

    datasets = {
        "dataset_I": ("Dataset I (gaussianas)", X1, y1, ("Classe A", "Classe B")),
        "dataset_II": ("Dataset II (cascas)", X2, y2, ("Classe C (núcleo)", "Classe D (casca)")),
    }

    # C — PCA e medidas geométricas, sempre nos dados 5D originais.
    explained = figure4(datasets)
    center_distance = {
        key: float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))
        for key, (_, X, y, _) in datasets.items()
    }
    radii = {key: np.linalg.norm(X, axis=1) for key, (_, X, _, _) in datasets.items()}
    figure5(datasets, radii)

    # D — uma função das entradas que separa o Dataset II.
    predicted_shell = radial_rule(X2) > 0
    rule_accuracy = float(np.mean(predicted_shell == (y2 == 1)))

    print("Exercise 2")
    for key in datasets:
        print(f"  {key}: distância entre centros = {center_distance[key]:.3f}, "
              f"PC1+PC2 = {100 * sum(explained[key]):.1f}%")
    print(f"  regra ||x||^2 > {THRESHOLD ** 2}: acurácia no Dataset II = {100 * rule_accuracy:.1f}%")

    radius_stats = {
        key: {
            str(c): {"mean": float(r[y == c].mean()), "min": float(r[y == c].min()), "max": float(r[y == c].max())}
            for c in (0, 1)
        }
        for key, (_, _, y, _) in datasets.items()
        for r in [radii[key]]
    }
    return {
        "min_eigenvalues": min_eigenvalues,
        "explained_variance": explained,
        "explained_variance_pc1_pc2": {key: float(sum(v)) for key, v in explained.items()},
        "center_distance_5d": center_distance,
        "center_distance_theory_dataset_I": float(np.linalg.norm(MU_B - MU_A)),
        "radius_stats": radius_stats,
        "radial_rule_threshold_squared": THRESHOLD**2,
        "radial_rule_accuracy_dataset_II": rule_accuracy,
    }
