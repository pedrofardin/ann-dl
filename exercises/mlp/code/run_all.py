"""Roda os exercícios em ordem, com um único gerador aleatório, e salva os números.

Uso, a partir da raiz do repositório:

    python docs/exercises/mlp/code/run_all.py

As figuras vão para docs/exercises/mlp/figures/ e os números para code/results.json.
"""

import json

import numpy as np

import ex2_binary
import ex3_multiclass
import ex4_depth
from common import CODE_DIR, SEED


def main():
    rng = np.random.default_rng(SEED)  # o mesmo gerador em todo o relatório
    results = {"exercise2": ex2_binary.run(rng)}
    results["exercise3"] = ex3_multiclass.run(rng)
    shallow_history = results["exercise3"].pop("history")  # vai para a Figura 6, não para o JSON
    results["exercise4"] = ex4_depth.run(rng, shallow_history)
    (CODE_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Números salvos em {CODE_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
