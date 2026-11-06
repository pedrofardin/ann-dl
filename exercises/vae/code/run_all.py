"""Roda os dois exercícios em ordem e salva os números.

Uso, a partir da raiz do repositório:

    python docs/exercises/vae/code/run_all.py

O MNIST é baixado em data/ (fora do Git). As figuras vão para docs/exercises/vae/figures/ e os
números para code/results.json.
"""

import json

import ex1_mnist
import ex2_latent2d
from common import CODE_DIR, load_mnist


def main():
    data = load_mnist()
    model16, results1 = ex1_mnist.run(data)
    results = {"exercise1": results1, "exercise2": ex2_latent2d.run(data, model16, results1)}
    (CODE_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Números salvos em {CODE_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
