"""Perceptron de camada única, escrito do zero com NumPy.

Esta é a única implementação do modelo no relatório. Os dois exercícios chamam a mesma
função train(); o Exercise 2 só liga o rastreio do pocket (pocket=True).
"""

import numpy as np


def step(z):
    """Ativação degrau: 1 se z >= 0 e 0 caso contrário. Aceita um número ou um vetor."""
    return np.where(z >= 0, 1, 0)


def predict(X, w, b):
    """Predição y_hat = step(w·x + b) para cada linha de X."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fração de amostras de X com predição igual ao rótulo."""
    return float(np.mean(predict(X, w, b) == y))


def init_weights(rng):
    """w ~ N(0, 0.01²) em cada coordenada e b = 0, como pede o enunciado. Nunca w = 0."""
    return rng.normal(0, 0.01, size=2), 0.0


def train(X, y, w0, b0, eta, max_epochs=100, pocket=False):
    """Treina o perceptron com a regra dirigida pelo erro, para rótulos em {0, 1}.

    Para cada amostra (x, y), na ordem de X:
        erro = y - y_hat            (0 se acertou; +1 ou -1 se errou)
        w <- w + eta * erro * x
        b <- b + eta * erro

    Para quando uma época inteira não faz nenhuma atualização, ou depois de max_epochs épocas.
    Com pocket=True, guarda uma cópia de (w, b) toda vez que uma atualização leva a acurácia no
    dataset completo acima da melhor já vista. É a única diferença entre os dois modos.
    """
    w, b = np.array(w0, dtype=float), float(b0)  # cópias: não altera o w0 de quem chamou

    history = {"accuracy": [], "updates": [], "pocket_accuracy": []}
    best = {"w": w.copy(), "b": b, "accuracy": accuracy(X, y, w, b), "epoch": 0, "update": 0}
    total_updates = 0

    for epoch in range(1, max_epochs + 1):
        updates = 0
        for xi, yi in zip(X, y):
            error = yi - step(xi @ w + b)
            if error == 0:  # acertou: a regra não muda nada
                continue
            w += eta * error * xi
            b += eta * error
            updates += 1
            total_updates += 1

            if pocket:  # algoritmo pocket: guarda o melhor (w, b) visto até agora
                acc = accuracy(X, y, w, b)
                if acc > best["accuracy"]:
                    best = {"w": w.copy(), "b": b, "accuracy": acc,
                            "epoch": epoch, "update": total_updates}

        # Acurácia no dataset completo depois de cada época.
        history["accuracy"].append(accuracy(X, y, w, b))
        history["updates"].append(updates)
        if pocket:
            history["pocket_accuracy"].append(best["accuracy"])
        if updates == 0:  # uma passagem sem erro: convergiu
            break

    result = {
        "w": w, "b": b,
        "accuracy": accuracy(X, y, w, b),
        "epochs": len(history["updates"]),
        "converged": history["updates"][-1] == 0,
        "total_updates": total_updates,
        "history": history,
    }
    if pocket:
        result["pocket"] = best
    return result
