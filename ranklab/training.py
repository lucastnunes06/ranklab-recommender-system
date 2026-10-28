"""Comparação de modelos com seleção apenas na validação por sessão.

IDs são sequenciais, mas não representam datas reais. Nenhum score do teste
participa da escolha de hiperparâmetros.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .metrics import (
    calibration_diagnostics,
    evaluate,
    paired_bootstrap_mean_difference,
    session_metrics,
)
from .pairwise import fit_pairwise
from .synthetic import CAT, FEATURES, NUM, make_sessions

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

# Definido antes do teste. A ordem desempata por configuração mais simples.
HGB_CANDIDATES = (
    {"max_iter": 80, "max_depth": 3, "learning_rate": 0.08},
    {"max_iter": 125, "max_depth": 3, "learning_rate": 0.08},
    {"max_iter": 125, "max_depth": 4, "learning_rate": 0.08},
)


def make_hgb(parameters: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("numeric", "passthrough", NUM),
        ])),
        ("ranker", HistGradientBoostingClassifier(
            **parameters, l2_regularization=2.0, random_state=seed,
        )),
    ])


def make_logistic(seed: int) -> Pipeline:
    return Pipeline([
        ("preprocessing", ColumnTransformer([
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
            ("numeric", StandardScaler(), NUM),
        ])),
        ("ranker", LogisticRegression(max_iter=600, random_state=seed)),
    ])


def split_sessions(frame, sessions: int):
    """Particionamento sem misturar candidatos da mesma sessão."""
    end_train, end_validation = int(0.6 * sessions), int(0.8 * sessions)
    training = frame[frame["session_id"] < end_train].copy()
    validation = frame[(frame["session_id"] >= end_train) & (frame["session_id"] < end_validation)].copy()
    testing = frame[frame["session_id"] >= end_validation].copy()
    return training, validation, testing


def compare(seed: int = 17, sessions: int = 1450):
    """Treina modelos para análise, sem escrever arquivos.

    Retorna o modelo selecionado exclusivamente pela validação.
    """
    frame, catalog = make_sessions(n_sessions=sessions, seed=seed)
    train_frame, val, test = split_sessions(frame, sessions)
    options = []
    for idx, params in enumerate(HGB_CANDIDATES):
        candidate = make_hgb(params, seed)
        candidate.fit(train_frame[FEATURES], train_frame["clicked"])
        predicted = val.copy()
        predicted["score"] = candidate.predict_proba(val[FEATURES])[:, 1]
        quality = evaluate(predicted)
        options.append({
            "id": f"hgb_{idx + 1}", "params": params,
            "validation_ndcg_at_5": quality["ndcg_at_5"],
            "model": candidate,
        })

    # Somente a validação escolhe o modelo e seus parâmetros. Não consulta o teste.
    best = max(options, key=lambda item: item["validation_ndcg_at_5"])
    selected = best["model"]
    logistic = make_logistic(seed)
    logistic.fit(train_frame[FEATURES], train_frame["clicked"])
    pairwise = fit_pairwise(train_frame, seed=seed)

    for group in (val, test):
        group["score"] = selected.predict_proba(group[FEATURES])[:, 1]
        group["logistic_score"] = logistic.predict_proba(group[FEATURES])[:, 1]
        group["pairwise_score"] = pairwise.score(group)
        group["rating_score"] = group["rating"]
        group["fast_score"] = -group["eta_minutes"]

    source = test
    metrics = {
        "dataset": "100% sintético: preferências, restaurantes e cliques inventados",
        "seed": seed,
        "split": "IDs de sessão sequenciais, sem timestamps reais; NÃO é um split temporal validado",
        "train_sessions": int(train_frame["session_id"].nunique()),
        "validation_sessions": int(val["session_id"].nunique()),
        "test_sessions": int(test["session_id"].nunique()),
        "candidates_per_session": 12,
        "selection": {
            "criterion": "NDCG@5 na validação; desempate pela ordem predefinida da busca",
            "selected_id": best["id"], "selected_parameters": best["params"],
            "candidates": [
                {key: value for key, value in option.items() if key != "model"}
                for option in options
            ],
            "test_not_used_for_selection": True,
        },
        "test": {
            "model": evaluate(test),
            "logistic_baseline": evaluate(test, "logistic_score"),
            "pairwise_baseline": evaluate(test, "pairwise_score"),
            "rating_baseline": evaluate(test, "rating_score"),
            "fastest_baseline": evaluate(test, "fast_score"),
        },
        "validation": {
            "model": evaluate(val),
            "logistic_baseline": evaluate(val, "logistic_score"),
            "pairwise_baseline": evaluate(val, "pairwise_score"),
            "rating_baseline": evaluate(val, "rating_score"),
            "fastest_baseline": evaluate(val, "fast_score"),
        },
        "calibration": {
            "test_synthetic": calibration_diagnostics(test["clicked"], test["score"]),
            "validation_synthetic": calibration_diagnostics(val["clicked"], val["score"]),
        },
        "limitations": "Avaliação offline e artificial; resultados não equivalem a CTR, vendas, satisfação ou calibração em pessoas reais.",
    }
    model_ndcg = session_metrics(source).set_index("session_id")["ndcg_at_5"]
    metrics["paired_bootstrap_ndcg_difference"] = {}
    for key, score_col in (
        ("rating_baseline", "rating_score"),
        ("fastest_baseline", "fast_score"),
        ("logistic_baseline", "logistic_score"),
        ("pairwise_baseline", "pairwise_score"),
    ):
        compared = session_metrics(source, score_col).set_index("session_id")["ndcg_at_5"]
        metrics["paired_bootstrap_ndcg_difference"][key] = paired_bootstrap_mean_difference(
            model_ndcg.values, compared.loc[model_ndcg.index].values, seed=seed + 4000,
        )
    return metrics, selected, test, catalog


def train(seed: int = 17, sessions: int = 1450) -> dict:
    metrics, selected, test, catalog = compare(seed=seed, sessions=sessions)
    ART.mkdir(exist_ok=True)
    joblib.dump(selected, ART / "ranker.joblib")
    (ART / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "data").mkdir(exist_ok=True)
    catalog.to_csv(ROOT / "data" / "fictional_restaurants.csv", index=False)
    test.drop(columns=["logistic_score", "pairwise_score", "rating_score", "fast_score"]).head(60).to_csv(
        ROOT / "data" / "holdout_sample.csv", index=False
    )
    return metrics


def load_ranker():
    path = ART / "ranker.joblib"
    if not path.exists():
        raise FileNotFoundError("Modelo indisponível. Execute: python scripts/train.py")
    # joblib/pickle só devem ser carregados de fontes confiáveis.
    return joblib.load(path)
