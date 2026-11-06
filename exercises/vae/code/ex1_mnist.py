"""Exercício 1 — Um VAE no MNIST, com d = 16.

Treina o modelo, calcula os números de teste e produz as Figuras 1, 2 e 3. Também mede os
números que sustentam as respostas do item 1E.
"""

import matplotlib.pyplot as plt
import torch

from common import loss_curves_figure, save, tile
from vae import evaluate, kl_per_dim, loss_terms, sample_prior, train

D = 16


def first_of_each_digit(y):
    """Índice da primeira imagem de cada dígito 0–9 no conjunto."""
    return [int((y == k).nonzero()[0]) for k in range(10)]


def figure2(model, x_test, idx):
    """Figura 2: original (em cima) e reconstrução decodificando μ (embaixo)."""
    x = x_test[idx]
    with torch.no_grad():
        mu, _ = model.encode(x)
        x_hat = model.decode(mu)  # decodifica μ: nada de sortear z
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.9), layout="constrained")
    for k in range(10):
        for row, img in enumerate([x[k], x_hat[k]]):
            ax = axes[row, k]
            ax.imshow(img.view(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, k].set_title(f"{k} (#{idx[k]})", fontsize=9)
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon. de μ")
    fig.suptitle(f"Figura 2 — Primeira imagem de teste de cada dígito e sua reconstrução (d = {D})")
    fig.supxlabel("Dígito (índice no conjunto de teste)")
    save(fig, "fig2_reconstructions.png")


def figure3(samples):
    """Figura 3: grade 8 × 8 de imagens decodificadas de z ~ N(0, I)."""
    fig, ax = plt.subplots(figsize=(7, 7.3), layout="constrained")
    ax.imshow(tile(samples, 8, 8), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([28 * c + 13.5 for c in range(8)], range(1, 9))
    ax.set_yticks([28 * r + 13.5 for r in range(8)], range(1, 9))
    ax.set_xlabel("Coluna da grade")
    ax.set_ylabel("Linha da grade")
    ax.set_title(f"Figura 3 — 64 amostras do prior: x̂ = decoder(z), z ~ N(0, I), d = {D}")
    save(fig, "fig3_samples.png")


def gray_fraction(x):
    """Fração dos pixels com valor em (0.1, 0.9): nem apagados nem acesos."""
    return ((x > 0.1) & (x < 0.9)).float().mean().item()


@torch.no_grad()
def analysis_numbers(model, x_val, x_test, samples):
    """Os números do item 1E."""
    mu, logvar = model.encode(x_test)
    kl_dims = kl_per_dim(mu, logvar)

    # Onde as amostras do prior caem: distância de cada z ~ N(0, I) ao μ de teste mais próximo,
    # comparada com a distância de um μ de validação ao μ de teste mais próximo. Só nas dimensões
    # ativas: nas outras, μ ≈ 0 e o decoder ignora a coordenada, que só somaria ruído do prior.
    active = kl_dims > 0.1
    g = torch.Generator().manual_seed(1)
    z_prior = torch.randn(len(x_test), D, generator=g)
    mu_val, _ = model.encode(x_val)
    nn_prior = torch.cdist(z_prior[:, active], mu[:, active]).min(dim=1).values
    nn_val = torch.cdist(mu_val[:, active], mu[:, active]).min(dim=1).values

    # O que o decoder devolve: probabilidades, não pixels 0/1.
    x_hat_mu = model.decode(mu)
    rec_mu = loss_terms(x_test, x_hat_mu, mu, logvar)[0].item()

    return {
        "kl_per_dim_test": [round(v, 3) for v in kl_dims.tolist()],
        "active_dims_kl_gt_0.1": int(active.sum()),
        "mean_posterior_sigma_test": torch.exp(0.5 * logvar).mean().item(),
        "aggregate_var_per_dim_mean": (mu.var(dim=0) + logvar.exp().mean(dim=0)).mean().item(),
        "nn_dist_prior_to_test_mu_median": nn_prior.median().item(),
        "nn_dist_val_mu_to_test_mu_median": nn_val.median().item(),
        "frac_prior_farther_than_p95_val": (nn_prior > nn_val.quantile(0.95)).float().mean().item(),
        "gray_fraction_test_images": gray_fraction(x_test),
        "gray_fraction_recon_mu": gray_fraction(x_hat_mu),
        "gray_fraction_prior_samples": gray_fraction(samples),
        "test_rec_decoding_mu": rec_mu,
    }


def run(data):
    (x_train, _), (x_val, _), (x_test, y_test) = data["train"], data["val"], data["test"]
    model, history = train(D, x_train, x_val)

    # Daqui para baixo o treino acabou: o teste entra só para os números finais e as figuras.
    test = evaluate(model, x_test)
    loss_curves_figure(history, f"Figura 1 — As duas parcelas da perda, treino e validação (d = {D})",
                       "fig1_loss_terms.png")

    idx = first_of_each_digit(y_test)
    figure2(model, x_test, idx)

    torch.manual_seed(0)  # semente do enunciado para a Figura 3
    samples = sample_prior(model, 64)
    figure3(samples)

    results = {
        "d": D,
        "test": test,
        "history": history,
        "val_kl_first_epoch": history["val_kl"][0],
        "val_kl_last_epoch": history["val_kl"][-1],
        "figure2_test_indices": idx,
        "analysis": analysis_numbers(model, x_val, x_test, samples),
    }
    return model, results
