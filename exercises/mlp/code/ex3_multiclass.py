"""Exercício 3 — Multiclasse, a mesma rede.

Gera os dados, ajusta a linha de base linear, treina o MLP com saída softmax e produz as
Figuras 4 e 5. A classe MLP é a mesma do Exercício 2, sem nenhuma mudança: só os argumentos
mudam (tamanhos das camadas, output="softmax", taxa de aprendizado e épocas).

Também recupera a estrutura de grupos dos dados (qual dos 6 grupos gerou cada amostra), que
o item D usa para explicar a confusão entre classes.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from common import loss_figure, save
from mlp import MLP

# Parâmetros do enunciado, sem alterações.
DATA = dict(n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
            n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42)
N_CLUSTERS = 6

# Hiperparâmetros escolhidos para o MLP. O Exercício 4 reusa ETA e EPOCHS.
LAYERS = [4, 16, 3]  # 4 entradas, 1 camada oculta de 16 neurônios tanh, 3 saídas softmax
ETA = 0.1
EPOCHS = 2000


def generate():
    return make_classification(**DATA)


def split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def cluster_structure(X):
    """Recupera de qual grupo veio cada amostra de X e o vértice do hipercubo de cada grupo.

    O make_classification sorteia tudo e só no fim embaralha as linhas e as colunas. Com
    shuffle=False, os sorteios anteriores são os mesmos: saem as mesmas amostras, sem embaralhar.
    Nessa ordem, o grupo k ocupa as linhas [250k, 250(k+1)), tem rótulo k % 3 (exceto o ~1% de
    rótulos que flip_y=0.01 sorteia de novo) e foi deslocado para um vértice de {-1, +1}^4.
    """
    X_raw, _ = make_classification(**DATA, shuffle=False)
    raw_cluster = np.repeat(np.arange(N_CLUSTERS), len(X_raw) // N_CLUSTERS)

    # As colunas foram permutadas: acha a coluna de X_raw com os mesmos valores de cada coluna de X.
    cols = [int(np.argmin([np.abs(np.sort(X[:, j]) - np.sort(X_raw[:, i])).max() for i in range(4)]))
            for j in range(X.shape[1])]
    X_raw = X_raw[:, cols]

    # As linhas foram permutadas: casa cada amostra de X com a sua cópia em X_raw.
    index = {row.tobytes(): k for row, k in zip(np.round(X_raw, 10), raw_cluster)}
    cluster = np.array([index[row.tobytes()] for row in np.round(X, 10)])

    # O centro de cada grupo é um vértice: o sinal da média recupera as coordenadas ±1.
    vertices = np.array([np.sign(X[cluster == k].mean(axis=0)) for k in range(N_CLUSTERS)]).astype(int)
    return cluster, vertices


def figure5(cm, acc):
    """Figura 5: matriz de confusão no teste, em contagens."""
    fig, ax = plt.subplots(figsize=(6, 5.2), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_yticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    fig.colorbar(im, ax=ax, label="Amostras de teste")
    ax.set_title(f"Figura 5 — Matriz de confusão no teste\nMLP {LAYERS}, acurácia = {100 * acc:.1f}%")
    save(fig, "fig5_confusion.png")


def run(rng):
    # A — dados e divisão 80/20 estratificada, com a mesma semente do Exercício 2.
    X, y = generate()
    X_train, X_test, y_train, y_test = split(X, y)

    # Linha de base linear (pedida no item D).
    baseline_acc = float(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))

    # B/C — a mesma classe MLP, agora com 3 saídas, softmax e entropia cruzada categórica.
    net = MLP(LAYERS, rng, hidden="tanh", output="softmax")
    history = net.fit(X_train, y_train, ETA, EPOCHS)
    train_acc, test_acc = net.accuracy(X_train, y_train), net.accuracy(X_test, y_test)
    pred = net.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])

    loss_figure({f"MLP {LAYERS}, $\\eta$ = {ETA}": history},
                "Figura 4 — Perda de treino × época (Exercício 3)", "fig4_loss.png")
    figure5(cm, test_acc)

    # D — o par mais confundido: soma dos dois sentidos fora da diagonal.
    pairs = {f"{a}-{b}": int(cm[a, b] + cm[b, a]) for a, b in [(0, 1), (0, 2), (1, 2)]}

    # D — estrutura de grupos: vértice de cada grupo e distâncias entre grupos de classes diferentes.
    cluster, vertices = cluster_structure(X)
    cluster_class = [k % 3 for k in range(N_CLUSTERS)]
    hamming = (vertices[:, None, :] != vertices[None, :, :]).sum(axis=2)
    _, test_idx, _, _ = split(np.arange(len(y)), y)  # mesma divisão, aplicada aos índices
    assert np.array_equal(X[test_idx], X_test)
    test_cluster = cluster[test_idx]
    errors_by_cluster = {
        f"grupo {k} (classe {cluster_class[k]})": {
            "n_test": int((test_cluster == k).sum()),
            "erros": int(((test_cluster == k) & (pred != y_test)).sum()),
            "previstos_como": np.bincount(pred[(test_cluster == k) & (pred != y_test)], minlength=3).tolist(),
        }
        for k in range(N_CLUSTERS)
    }

    print("Exercício 3")
    print(f"  linha de base linear: teste = {100 * baseline_acc:.2f}%")
    print(f"  MLP {LAYERS}, eta = {ETA}, {EPOCHS} épocas: perda final = {history[-1]:.4f}, "
          f"treino = {100 * train_acc:.2f}%, teste = {100 * test_acc:.2f}%")
    print(f"  confusões por par (dois sentidos): {pairs}")

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "baseline_test_accuracy": baseline_acc,
        "mlp": {
            "layers": LAYERS,
            "hidden_activation": "tanh",
            "output": "softmax + entropia cruzada categórica",
            "n_params": net.n_params(),
            "eta": ETA,
            "epochs": EPOCHS,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        },
        "confusion_matrix": cm.tolist(),
        "confusions_per_pair": pairs,
        "clusters": {
            "class_of_cluster": cluster_class,
            "vertices": vertices.tolist(),
            "hamming_between_clusters": hamming.tolist(),
            "labels_flipped": int((y != np.array(cluster_class)[cluster]).sum()),
            "test_errors_by_cluster": errors_by_cluster,
        },
        "history": history,  # o Exercício 4 desenha esta curva na Figura 6
    }
