"""Confere a passagem reversa contra diferenças finitas, para as duas saídas e para 1 e 2 camadas ocultas.

Para cada parâmetro theta, a derivada numérica é

    (L(theta + e) - L(theta - e)) / (2e)

e o erro relativo contra o gradiente de backward() deve ficar perto de 1e-7 ou menos.
Não faz parte do relatório: usa o próprio gerador, para não mudar a sequência de sorteios de run_all.py.

Uso: python docs/exercises/mlp/code/gradcheck.py
"""

import numpy as np

from mlp import MLP


def numeric_grads(net, X, y, e=1e-6):
    grads = []
    for W, b in zip(net.W, net.b):
        layer = []
        for theta in (W, b):
            g = np.zeros_like(theta)
            for i in np.ndindex(theta.shape):
                old = theta[i]
                theta[i] = old + e
                plus = net.loss(X, y)
                theta[i] = old - e
                minus = net.loss(X, y)
                theta[i] = old
                g[i] = (plus - minus) / (2 * e)
            layer.append(g)
        grads.append(layer)
    return grads


def main():
    rng = np.random.default_rng(0)
    cases = [
        ([2, 5, 1], "sigmoid", 2),
        ([4, 6, 3], "softmax", 3),
        ([4, 6, 5, 3], "softmax", 3),
    ]
    for sizes, output, k in cases:
        net = MLP(sizes, rng, output=output)
        X = rng.normal(size=(20, sizes[0]))
        y = rng.integers(0, k if output == "softmax" else 2, size=20)
        analytic = net.backward(net.forward(X), net.targets(y))
        numeric = numeric_grads(net, X, y)
        worst = 0.0
        for (dW, db), (nW, nb) in zip(analytic, numeric):
            for a, n in ((dW, nW), (db, nb)):
                worst = max(worst, np.linalg.norm(a - n) / (np.linalg.norm(a) + np.linalg.norm(n)))
        status = "ok" if worst < 1e-6 else "FALHOU"
        print(f"{str(sizes):14s} {output:8s} erro relativo máximo = {worst:.2e}  {status}")


if __name__ == "__main__":
    main()
