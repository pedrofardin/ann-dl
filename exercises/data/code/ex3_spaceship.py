"""Exercise 3 — Preparing real-world data for a neural network (Spaceship Titanic).

Descreve o train.csv, separa treino e teste ANTES de qualquer estatística e prepara as
features para uma rede com ativação tanh. Produz a figura auxiliar do item C e a Figura 6.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DATA_DIR, SEED, save

TARGET = "Transported"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age", *SPEND]
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROPPED = ["Cabin", "Name", "PassengerId"]
LOG_COLUMNS = [*SPEND, "TotalSpend"]
SCALED = ["Age", *LOG_COLUMNS]

CLASS_NAMES = {0: "Transported = False", 1: "Transported = True"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:orange"}


def describe(df):
    """A — balanço do alvo, valores ausentes e estatísticas dos gastos (arquivo train.csv inteiro)."""
    missing = df.isna().sum()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "balance": {str(k): float(v) for k, v in df[TARGET].value_counts(normalize=True).items()},
        "missing": {
            col: {"count": int(missing[col]), "percent": float(100 * missing[col] / len(df))}
            for col in df.columns
        },
        "spend_stats": {
            col: {"mean": float(df[col].mean()), "median": float(df[col].median()), "max": float(df[col].max())}
            for col in SPEND
        },
    }


class TanhPreprocessor:
    """Aprende todas as estatísticas no treino (fit) e aplica as mesmas ao teste (transform)."""

    def fit(self, X):
        X = X.drop(columns=DROPPED)
        # Numéricas: mediana do treino. Os gastos têm cauda longa; a média seria puxada pelos extremos.
        self.num_imputer = SimpleImputer(strategy="median").fit(X[NUMERIC])
        # Categóricas: categoria mais frequente do treino.
        self.cat_imputer = SimpleImputer(strategy="most_frequent").fit(X[CATEGORICAL].astype(object))
        X = self._impute_and_engineer(X)
        self.scaler = StandardScaler().fit(X[SCALED])
        # handle_unknown="ignore": categoria nova no teste vira um vetor só de zeros, sem erro.
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[CATEGORICAL])
        return self

    def _impute_and_engineer(self, X):
        """Imputação, TotalSpend e log(1 + x). Usa só objetos já ajustados no treino."""
        X = X.copy()
        X[NUMERIC] = self.num_imputer.transform(X[NUMERIC])
        X[CATEGORICAL] = self.cat_imputer.transform(X[CATEGORICAL].astype(object)).astype(str)
        X["TotalSpend"] = X[SPEND].sum(axis=1)  # soma depois da imputação, para não somar NaN
        X[LOG_COLUMNS] = np.log1p(X[LOG_COLUMNS])  # log(1 + x) comprime a cauda longa dos gastos
        return X

    def transform(self, X):
        X = self._impute_and_engineer(X.drop(columns=DROPPED))
        scaled = self.scaler.transform(X[SCALED])  # média 0 e desvio 1, com a média e o desvio do treino
        onehot = self.encoder.transform(X[CATEGORICAL])
        columns = [*SCALED, *self.encoder.get_feature_names_out(CATEGORICAL)]
        return pd.DataFrame(np.hstack([scaled, onehot]), columns=columns, index=X.index)


def hist_by_class(ax, values, y, bins):
    """Histograma sobreposto das duas classes do alvo."""
    for c in (0, 1):
        ax.hist(values[y == c].dropna(), bins=bins, alpha=0.6, color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    ax.set_ylabel("Número de passageiros")
    ax.legend(loc="upper right")


def figure_before_after(before, after, y, name, title, after_label):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, values, label in [(axes[0], before, "valor original"), (axes[1], after, after_label)]:
        hist_by_class(ax, values, y, np.histogram_bin_edges(values.dropna(), bins=50))
        ax.set_xlabel(f"{values.name} — {label}")
    axes[0].set_title("Antes")
    axes[1].set_title("Depois")
    fig.suptitle(title)
    save(fig, name)


def run():
    df = pd.read_csv(DATA_DIR / "train.csv")

    # A — conhecer os dados.
    description = describe(df)

    # B — separar ANTES de calcular qualquer estatística de transformação.
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # C — pré-processar: fit só no treino, transform no treino e no teste.
    prep = TanhPreprocessor().fit(X_train)
    Xtr = prep.transform(X_train)
    Xte = prep.transform(X_test)

    # Categoria que não existe no treino: HomePlanet = "Pluto".
    unknown = X_train.iloc[[0]].copy()
    unknown["HomePlanet"] = "Pluto"
    unknown_encoded = prep.transform(unknown)
    unknown_onehot = {c: float(unknown_encoded[c].iloc[0]) for c in unknown_encoded if c.startswith("HomePlanet_")}

    log_example = np.log1p(X_train["RoomService"]).rename("RoomService")
    figure_before_after(X_train["RoomService"], log_example, y_train, "fig_c_log1p_roomservice.png",
                        "Figura auxiliar (item C) — RoomService antes e depois de log(1 + x), treino",
                        "log(1 + x)")

    # D — verificar e visualizar.
    figure_before_after(X_train["FoodCourt"], Xtr["FoodCourt"], y_train, "fig6_foodcourt.png",
                        "Figura 6 — FoodCourt antes e depois do pré-processamento, treino",
                        "imputado, log(1 + x) e padronizado")

    # Comparação: gastos imputados e padronizados SEM log(1 + x). Só mede o efeito do log;
    # não entra na matriz final.
    medians = dict(zip(NUMERIC, prep.num_imputer.statistics_))
    raw_spend = X_train[SPEND].fillna({col: medians[col] for col in SPEND})
    z_without_log = (raw_spend - raw_spend.mean()) / raw_spend.std(ddof=0)

    checks = {
        "max_z_without_log_train": {c: float(z_without_log[c].max()) for c in SPEND},
        "share_abs_above_3_without_log_train": float((z_without_log.abs() > 3).to_numpy().mean()),
        "nan_train": int(Xtr.isna().sum().sum()),
        "nan_test": int(Xte.isna().sum().sum()),
        "shape_train": list(Xtr.shape),
        "shape_test": list(Xte.shape),
        "min_train": float(Xtr.to_numpy().min()),
        "max_train": float(Xtr.to_numpy().max()),
        "min_test": float(Xte.to_numpy().min()),
        "max_test": float(Xte.to_numpy().max()),
        "scaled_mean_train": {c: float(Xtr[c].mean()) for c in SCALED},
        "scaled_std_train": {c: float(Xtr[c].std(ddof=0)) for c in SCALED},
        "scaled_min_max_train": {c: [float(Xtr[c].min()), float(Xtr[c].max())] for c in SCALED},
        "scaled_min_max_test": {c: [float(Xte[c].min()), float(Xte[c].max())] for c in SCALED},
        "share_abs_above_3_train": float((Xtr[SCALED].abs() > 3).to_numpy().mean()),
    }

    print("Exercise 3")
    print(f"  treino {Xtr.shape}, teste {Xte.shape}, NaN = {checks['nan_train']} / {checks['nan_test']}")
    print(f"  faixa treino [{checks['min_train']:.3f}, {checks['max_train']:.3f}], "
          f"teste [{checks['min_test']:.3f}, {checks['max_test']:.3f}]")

    return {
        "description": description,
        "split": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "positive_share_train": float(y_train.mean()),
            "positive_share_test": float(y_test.mean()),
        },
        "foodcourt_train_before": {
            "mean": float(X_train["FoodCourt"].mean()),
            "median": float(X_train["FoodCourt"].median()),
        },
        "imputation_values": {
            "numeric_median": dict(zip(NUMERIC, prep.num_imputer.statistics_.tolist())),
            "categorical_mode": dict(zip(CATEGORICAL, map(str, prep.cat_imputer.statistics_))),
        },
        "features": list(Xtr.columns),
        "unknown_category_onehot": unknown_onehot,
        "checks": checks,
    }
