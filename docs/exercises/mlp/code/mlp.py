"""MLP (perceptron de múltiplas camadas) escrito do zero com NumPy.

Esta é a única implementação do modelo no relatório. Os Exercícios 2, 3 e 4 usam a mesma
classe MLP sem mudar uma linha: o que muda entre eles são só os argumentos.

- layer_sizes: os tamanhos das camadas, da entrada à saída. [2, 8, 1] é uma rede com 2
  entradas, 1 camada oculta de 8 neurônios e 1 saída; [4, 16, 16, 3] tem 2 camadas ocultas.
- output: "sigmoid" (saída binária + entropia cruzada binária) ou
  "softmax" (saída multiclasse + entropia cruzada categórica).

Notação (a mesma do Exercício 1): a camada l tem pesos W[l], de forma (n_saída, n_entrada), e
bias b[l]. Para um lote X com N linhas:

    u = h_anterior @ W[l].T + b[l]       pré-ativação
    h = g(u)                             ativação (tanh nas ocultas; sigmoide/softmax na saída)
"""

import numpy as np

EPS = 1e-12  # evita log(0) na perda


# --- Ativações das camadas ocultas -------------------------------------------------------
# Cada entrada guarda a função e a derivada. A derivada é escrita em função da SAÍDA h = g(u),
# que a passagem direta já guardou: tanh'(u) = 1 - tanh²(u) = 1 - h².

def tanh(u):
    return np.tanh(u)


def tanh_grad(h):
    return 1.0 - h ** 2


def relu(u):
    return np.maximum(u, 0.0)


def relu_grad(h):
    return (h > 0).astype(float)


HIDDEN = {"tanh": (tanh, tanh_grad), "relu": (relu, relu_grad)}


# --- Camadas de saída e perdas --------------------------------------------------------------

def sigmoid(u):
    """1 / (1 + e^-u), sem estouro numérico para |u| grande."""
    e = np.exp(-np.abs(u))
    return np.where(u >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def softmax(u):
    """e^u_k / sum_j e^u_j em cada linha. Subtrair o máximo não muda o resultado e evita estouro."""
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def binary_cross_entropy(p, Y):
    """L = -(1/N) sum [y log p + (1 - y) log(1 - p)]. p e Y têm forma (N, 1)."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))


def categorical_cross_entropy(P, Y):
    """L = -(1/N) sum_n sum_k y_nk log p_nk. P e Y (one-hot) têm forma (N, K)."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, EPS, 1.0)), axis=1)))


# Cada saída vem com a perda que forma par com ela. Nos dois pares, a derivada da perda em
# relação à pré-ativação da saída é a mesma: dL/du = (p - y) / N. É isso que deixa a passagem
# reversa idêntica para os casos binário e multiclasse.
OUTPUTS = {
    "sigmoid": (sigmoid, binary_cross_entropy),
    "softmax": (softmax, categorical_cross_entropy),
}


class MLP:
    """Rede totalmente conectada, de profundidade qualquer, treinada por gradiente descendente."""

    def __init__(self, layer_sizes, rng, hidden="tanh", output="sigmoid"):
        if len(layer_sizes) < 3:
            raise ValueError("o MLP precisa de pelo menos uma camada oculta: [entrada, oculta, ..., saída]")
        self.layer_sizes = list(layer_sizes)
        self.act, self.act_grad = HIDDEN[hidden]
        self.out, self.loss_fn = OUTPUTS[output]
        self.output = output

        # Inicialização de Xavier/Glorot: W ~ N(0, 2 / (n_entrada + n_saída)), b = 0.
        # Mantém a variância das ativações estável entre camadas, e o tanh fora da saturação.
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), size=(n_out, n_in)))
            self.b.append(np.zeros(n_out))

    def n_params(self):
        """Número total de pesos e biases."""
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    def targets(self, y):
        """Converte rótulos inteiros no alvo que a saída espera: coluna (N, 1) ou one-hot (N, K)."""
        y = np.asarray(y)
        if self.output == "sigmoid":
            return y.reshape(-1, 1).astype(float)
        return np.eye(self.layer_sizes[-1])[y]

    # --- Núcleo: passagem direta, passagem reversa e atualização ---------------------------

    def forward(self, X):
        """Passagem direta. Devolve a lista [X, h1, h2, ..., saída] com a saída de cada camada.

        A lista inteira fica guardada porque a passagem reversa precisa da entrada de cada camada.
        """
        hs = [X]
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            u = hs[-1] @ W.T + b
            hs.append(self.out(u) if l == last else self.act(u))
        return hs

    def backward(self, hs, Y):
        """Passagem reversa: devolve [(dL/dW[l], dL/db[l]) para cada camada l], da primeira à última.

        Começa pela saída, onde dL/du = (p - Y) / N (par sigmoide + BCE ou softmax + CCE), e
        encadeia para trás. Para a camada l, com entrada h_ant = hs[l] e delta = dL/du[l]:

            dL/dW[l] = delta.T @ h_ant          (soma sobre as N amostras do lote)
            dL/db[l] = soma de delta nas linhas
            dL/dh_ant = delta @ W[l]            (devolve o erro para a camada anterior)
            dL/du[l-1] = dL/dh_ant * g'(u[l-1]) (regra da cadeia pela ativação oculta)
        """
        N = Y.shape[0]
        delta = (hs[-1] - Y) / N
        grads = []
        for l in reversed(range(len(self.W))):
            grads.append((delta.T @ hs[l], delta.sum(axis=0)))
            if l > 0:  # a camada 0 recebe X, que não tem gradiente a propagar
                delta = (delta @ self.W[l]) * self.act_grad(hs[l])
        return grads[::-1]  # foi montada da última camada para a primeira

    def update(self, grads, eta):
        """Gradiente descendente: theta <- theta - eta * dL/dtheta, para cada peso e bias."""
        for l, (dW, db) in enumerate(grads):
            self.W[l] -= eta * dW
            self.b[l] -= eta * db

    # --- Treino e predição ------------------------------------------------------------------

    def loss(self, X, y):
        """Perda média no conjunto (X, y)."""
        return self.loss_fn(self.forward(X)[-1], self.targets(y))

    def fit(self, X, y, eta, epochs):
        """Treina por gradiente descendente em lote completo: 1 atualização por época.

        Devolve a perda de treino de cada época, medida na passagem direta antes da atualização.
        Sem sorteio dentro do treino: o resultado só depende da inicialização.
        """
        Y = self.targets(y)
        history = []
        for _ in range(epochs):
            hs = self.forward(X)
            history.append(self.loss_fn(hs[-1], Y))
            self.update(self.backward(hs, Y), eta)
        return history

    def predict_proba(self, X):
        """Saída da rede: P(classe 1), forma (N,), no caso binário; P(classe k), forma (N, K), no multiclasse."""
        p = self.forward(X)[-1]
        return p[:, 0] if self.output == "sigmoid" else p

    def predict(self, X):
        """Classe prevista: limiar 0.5 no caso binário; a classe de maior probabilidade no multiclasse."""
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int) if self.output == "sigmoid" else p.argmax(axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))
