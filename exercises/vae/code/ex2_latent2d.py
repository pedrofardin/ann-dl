"""Exercício 2 — Um espaço latente que dá para ver, com d = 2.

Treina o mesmo modelo do Exercício 1 com d = 2 e produz as Figuras 4, 5 e 6. A linha de baixo
da Figura 6 usa o modelo d = 16 do Exercício 1. Também mede os números do item 2E.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from common import save, tile
from vae import evaluate, train

D = 2
GRID = 15
Q = np.linspace(0.05, 0.95, GRID)  # quantis igualmente espaçados
Z_GRID = norm.ppf(Q)               # z = Φ⁻¹(q): cada célula tem a mesma massa do prior
INTERP_DIGITS = (0, 1)             # a Figura 6 vai da primeira imagem de teste de um 0 até a de um 1
STEPS = 10


def figure4(mu, y):
    """Figura 4: μ das 10 000 imagens de teste, coloridas pelo dígito."""
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 7.5), layout="constrained")
    for k in range(10):
        pts = mu[y == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.5, color=cmap(k), label=str(k), rasterized=True)
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Figura 4 — μ das 10 000 imagens de teste no espaço latente (d = 2)")
    ax.legend(title="Dígito", markerscale=5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)
    save(fig, "fig4_latent_scatter.png")


def figure5(model):
    """Figura 5: decodifica a grade 15 × 15 em z = Φ⁻¹(q) e monta as imagens na mesma disposição.

    A coluna é z₁, crescendo da esquerda para a direita; a linha é z₂, crescendo de baixo para
    cima. Assim a Figura 5 tem a mesma orientação da Figura 4.
    """
    z1, z2 = np.meshgrid(Z_GRID, Z_GRID[::-1])  # linha 0 (topo) = maior z₂
    z = torch.tensor(np.column_stack([z1.ravel(), z2.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        imgs = model.decode(z)
    fig, ax = plt.subplots(figsize=(9, 9.3), layout="constrained")
    ax.imshow(tile(imgs, GRID, GRID), cmap="gray", vmin=0, vmax=1)
    centers = [28 * c + 13.5 for c in range(GRID)]
    ax.set_xticks(centers, [f"{v:.2f}" for v in Z_GRID], rotation=90, fontsize=8)
    ax.set_yticks(centers, [f"{v:.2f}" for v in Z_GRID[::-1]], fontsize=8)
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$, q de 0.05 a 0.95")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$, q de 0.05 a 0.95")
    ax.set_title("Figura 5 — O plano latente decodificado, grade 15 × 15 nos quantis do prior (d = 2)")
    save(fig, "fig5_latent_grid.png")


def interpolation(model16, x_test, y_test):
    """As duas linhas da Figura 6: no espaço dos pixels e no espaço latente do modelo d = 16."""
    a, b = (int((y_test == k).nonzero()[0]) for k in INTERP_DIGITS)
    t = torch.linspace(0, 1, STEPS).unsqueeze(1)
    xa, xb = x_test[a], x_test[b]
    pixel_row = (1 - t) * xa + t * xb
    with torch.no_grad():
        mu, _ = model16.encode(torch.stack([xa, xb]))
        latent_row = model16.decode((1 - t) * mu[0] + t * mu[1])
    return a, b, t.squeeze(1), pixel_row, latent_row


def figure6(a, b, t, pixel_row, latent_row):
    """Figura 6: duas linhas de 10 imagens, t de 0 a 1."""
    fig, axes = plt.subplots(2, STEPS, figsize=(12, 3.1), layout="constrained")
    for k in range(STEPS):
        for row, imgs in enumerate([pixel_row, latent_row]):
            ax = axes[row, k]
            ax.imshow(imgs[k].view(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[1, k].set_xlabel(f"t = {t[k]:.2f}", fontsize=9)
    axes[0, 0].set_ylabel("Pixels")
    axes[1, 0].set_ylabel("Latente\n(d = 16)")
    fig.suptitle(f"Figura 6 — Interpolação entre as imagens de teste #{a} (dígito {INTERP_DIGITS[0]}) "
                 f"e #{b} (dígito {INTERP_DIGITS[1]})")
    save(fig, "fig6_interpolation.png")


@torch.no_grad()
def nearest_distance(x, ref, chunk=2000):
    """Distância euclidiana de cada linha de x à linha mais próxima de ref."""
    return torch.cat([torch.cdist(x[i:i + chunk], ref).min(dim=1).values
                      for i in range(0, len(x), chunk)])


@torch.no_grad()
def knn_confusion(mu_ref, y_ref, mu, y, k=15):
    """Matriz de confusão (10, 10) de um k-NN em μ: rótulos de referência da validação, alvo no teste.

    Mede quanto os dígitos se sobrepõem no espaço latente: onde os vizinhos de um ponto são de
    outro dígito, as nuvens se misturam.
    """
    pred = []
    for i in range(0, len(mu), 1000):
        idx = torch.cdist(mu[i:i + 1000], mu_ref).topk(k, largest=False).indices
        pred.append(torch.mode(y_ref[idx], dim=1).values)
    pred = torch.cat(pred)
    conf = torch.zeros(10, 10, dtype=torch.long)
    for true, p in zip(y.tolist(), pred.tolist()):
        conf[true, p] += 1
    return conf


def top_confused_pairs(conf, n=5):
    """Os n pares de dígitos {i, j} com mais confusões, somando os dois sentidos."""
    pairs = [((i, j), int(conf[i, j] + conf[j, i])) for i in range(10) for j in range(i + 1, 10)]
    return sorted(pairs, key=lambda p: -p[1])[:n]


@torch.no_grad()
def analysis_numbers(model2, model16, data, pixel_row, latent_row):
    """Os números do item 2E."""
    (x_train, _), (x_val, y_val), (x_test, y_test) = data["train"], data["val"], data["test"]
    results = {}
    for name, model in [("d2", model2), ("d16", model16)]:
        mu_val, _ = model.encode(x_val)
        mu_test, _ = model.encode(x_test)
        conf = knn_confusion(mu_val, y_val, mu_test, y_test)
        results[f"knn15_accuracy_{name}"] = conf.diag().sum().item() / len(y_test)
        results[f"top_confused_pairs_{name}"] = [[list(p), c] for p, c in top_confused_pairs(conf)]

    mu2, _ = model2.encode(x_test)
    # Mediana de μ de cada dígito, e onde ela cai na grade da Figura 5 (quantil q = Φ(z)).
    med = {k: mu2[y_test == k].median(dim=0).values.tolist() for k in range(10)}
    results["median_mu_d2"] = {k: [round(v, 3) for v in m] for k, m in med.items()}
    results["median_mu_d2_quantile"] = {k: [round(float(norm.cdf(v)), 3) for v in m]
                                        for k, m in med.items()}

    # Quão longe cada quadro da Figura 6 está da imagem de treino mais próxima (pixels).
    results["fig6_nn_train_dist_pixel_row"] = [round(v, 2) for v in nearest_distance(pixel_row, x_train).tolist()]
    results["fig6_nn_train_dist_latent_row"] = [round(v, 2) for v in nearest_distance(latent_row, x_train).tolist()]
    gray = lambda r: [round(((r[k] > 0.1) & (r[k] < 0.9)).float().mean().item(), 3) for k in range(len(r))]
    results["fig6_gray_fraction_pixel_row"] = gray(pixel_row)
    results["fig6_gray_fraction_latent_row"] = gray(latent_row)
    return results


def run(data, model16, results16):
    (x_train, _), (x_val, _), (x_test, y_test) = data["train"], data["val"], data["test"]
    model2, history = train(D, x_train, x_val)  # mesmos dados, semente, β e orçamento; só d muda

    test = evaluate(model2, x_test)
    with torch.no_grad():
        mu2, _ = model2.encode(x_test)
    figure4(mu2.numpy(), y_test.numpy())
    figure5(model2)
    a, b, t, pixel_row, latent_row = interpolation(model16, x_test, y_test)
    figure6(a, b, t, pixel_row, latent_row)

    return {
        "d": D,
        "test": test,
        "test_d16": results16["test"],
        "history": history,
        "interpolation_test_indices": [a, b],
        "interpolation_digits": list(INTERP_DIGITS),
        "analysis": analysis_numbers(model2, model16, data, pixel_row, latent_row),
    }
