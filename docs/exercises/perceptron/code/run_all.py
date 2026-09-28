"""Roda os dois exercícios em ordem, com um único gerador aleatório, e salva os números.

Uso, a partir da raiz do repositório:

    python docs/exercises/perceptron/code/run_all.py

As figuras vão para docs/exercises/perceptron/figures/ e os números para code/results.json.
"""

import json

import numpy as np

import ex1_separable
import ex2_overlapping
from common import CODE_DIR, SEED


def main():
    rng = np.random.default_rng(SEED)  # o mesmo gerador em todo o relatório
    results = {
        "exercise1": ex1_separable.run(rng),
        "exercise2": ex2_overlapping.run(rng),
    }
    (CODE_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Números salvos em {CODE_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
