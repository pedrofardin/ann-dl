"""O VAE do item 1B: modelo, reparametrização, as duas parcelas da perda, amostragem e treino.

O autograd do PyTorch calcula os gradientes. As partes próprias do VAE estão escritas à mão:
a reparametrização (`reparameterize`), a BCE somada nos pixels e a KL em forma fechada
(`loss_terms`) e a amostragem do prior (`sample_prior`). Não há classe de VAE de biblioteca
nem `torch.distributions`.
"""

import torch
import torch.nn.functional as F
from torch import nn

PIXELS = 784


class VAE(nn.Module):
    """Encoder 784 → 512 → 256 (ReLU), cabeças μ e log σ² 256 → d, decoder d → 256 → 512 → 784."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(PIXELS, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)  # a rede prevê log σ²: qualquer real vale, σ² > 0 sempre
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, PIXELS), nn.Sigmoid(),  # cada saída é a probabilidade do pixel estar "aceso"
        )

    def encode(self, x):
        """q(z | x) = N(μ, diag(σ²)): devolve μ e log σ², ambos (N, d)."""
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        """p(x | z): devolve x̂ (N, 784), a média de uma Bernoulli por pixel."""
        return self.decoder(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar, generator=None):
    """z = μ + σ ⊙ ε, com ε ~ N(0, I).

    O sorteio fica em ε, que não depende dos parâmetros. Assim z é uma função diferenciável
    de μ e de σ, e o gradiente da perda chega ao encoder. σ = exp(½ log σ²).
    """
    eps = torch.randn(mu.shape, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps


def loss_terms(x, x_hat, mu, logvar):
    """As duas parcelas da perda, em nats por imagem (soma nos pixels e nas dimensões, média no lote).

    Reconstrução: −Σ_j [x_j log x̂_j + (1 − x_j) log(1 − x̂_j)], a BCE somada nos 784 pixels.
    KL(q(z|x) ‖ N(0, I)) em forma fechada: ½ Σ_i (μ_i² + σ_i² − 1 − log σ_i²).
    """
    n = x.shape[0]
    rec = F.binary_cross_entropy(x_hat, x, reduction="sum") / n  # "sum", não "mean": ver o relatório
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / n
    return rec, kl


def kl_per_dim(mu, logvar):
    """KL média por imagem, separada por dimensão latente: um vetor (d,) cuja soma é a KL total."""
    return (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)).mean(dim=0)


def sample_prior(model, n, generator=None):
    """Gera n imagens: z ~ N(0, I) do prior e x̂ = decoder(z)."""
    z = torch.randn(n, model.d, generator=generator)
    with torch.no_grad():
        return model.decode(z)


@torch.no_grad()
def evaluate(model, x, beta=1.0, batch_size=1000, seed=0):
    """Reconstrução e KL médias num conjunto inteiro, com o z amostrado como no treino.

    O ruído ε vem de um gerador próprio, com semente fixa. Assim a avaliação dá o mesmo número
    sempre e não mexe no gerador global, que embaralha os lotes do treino.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    rec_sum, kl_sum = 0.0, 0.0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        x_hat, mu, logvar = model(xb, generator=g)
        rec, kl = loss_terms(xb, x_hat, mu, logvar)
        rec_sum += rec.item() * len(xb)
        kl_sum += kl.item() * len(xb)
    model.train()
    rec, kl = rec_sum / len(x), kl_sum / len(x)
    return {"rec": rec, "kl": kl, "neg_elbo": rec + beta * kl}


def train(d, x_train, x_val, beta=1.0, lr=1e-3, batch_size=128, epochs=20, seed=42):
    """Item 1C: Adam, η = 10⁻³, lotes de 128, 20 épocas.

    A cada época, grava as duas parcelas separadas: a média da época no treino e o valor no
    fim da época na validação. O teste não entra aqui.
    """
    torch.manual_seed(seed)  # antes de construir o modelo, como pede o enunciado
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(x_train))  # nova ordem a cada época (gerador global, semeado acima)
        rec_sum, kl_sum = 0.0, 0.0
        for i in range(0, len(x_train), batch_size):
            xb = x_train[perm[i:i + batch_size]]
            x_hat, mu, logvar = model(xb)
            rec, kl = loss_terms(xb, x_hat, mu, logvar)
            loss = rec + beta * kl  # a perda é a média no lote do −ELBO por imagem
            opt.zero_grad()
            loss.backward()
            opt.step()
            rec_sum += rec.item() * len(xb)  # pondera pelo tamanho: o último lote é menor
            kl_sum += kl.item() * len(xb)

        val = evaluate(model, x_val, beta)
        history["train_rec"].append(rec_sum / len(x_train))
        history["train_kl"].append(kl_sum / len(x_train))
        history["val_rec"].append(val["rec"])
        history["val_kl"].append(val["kl"])
        print(f"d={d} época {epoch:2d}: treino rec {history['train_rec'][-1]:7.2f} "
              f"KL {history['train_kl'][-1]:6.2f} | val rec {val['rec']:7.2f} KL {val['kl']:6.2f}")

    model.eval()
    return model, history
