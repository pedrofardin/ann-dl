"""Roda os três exercícios em ordem, com um único gerador aleatório, e salva os números.

Uso, a partir da raiz do repositório:

    python docs/exercises/data/code/run_all.py

As figuras vão para docs/exercises/data/figures/ e os números para code/results.json.
O Exercise 3 precisa de code/data/train.csv (Kaggle, competição Spaceship Titanic).
"""

import json

import numpy as np

import ex1_point_clouds
import ex2_nonlinearity
import ex3_spaceship
from common import CODE_DIR, SEED


def main():
    rng = np.random.default_rng(SEED)  # o mesmo gerador em todo o relatório
    results = {
        "exercise1": ex1_point_clouds.run(rng),
        "exercise2": ex2_nonlinearity.run(rng),
        "exercise3": ex3_spaceship.run(),  # não sorteia nada; o split usa random_state=SEED
    }
    (CODE_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Números salvos em {CODE_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
