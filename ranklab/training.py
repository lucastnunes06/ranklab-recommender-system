from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .synthetic import make_sessions, FEATURES, CAT, NUM
from .metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

def train(seed: int = 17, sessions: int = 1450) -> dict:
    frame, catalog = make_sessions(n_sessions=sessions, seed=seed)
    train_end, val_end = int(.6 * sessions), int(.8 * sessions)
    tr = frame[frame.session_id < train_end].copy()
    val = frame[(frame.session_id >= train_end) & (frame.session_id < val_end)].copy()
    test = frame[frame.session_id >= val_end].copy()
    pipeline = Pipeline([
        ("preprocessing", ColumnTransformer([
           ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
           ("numeric", "passthrough", NUM)])),
        ("ranker", HistGradientBoostingClassifier(max_iter=125, max_depth=4, learning_rate=.08,
                                                  l2_regularization=2.0, random_state=seed))
    ])
    pipeline.fit(tr[FEATURES], tr["clicked"])
    val["score"] = pipeline.predict_proba(val[FEATURES])[:, 1]
    val["rating_score"] = val["rating"]
    val["fast_score"] = -val["eta_minutes"]
    # Modelo escolhido usando APENAS a validação; teste nunca determina hiperparâmetros.
    validation = {"ml": evaluate(val), "rating": evaluate(val, "rating_score"), "fast": evaluate(val, "fast_score")}
    test["score"] = pipeline.predict_proba(test[FEATURES])[:, 1]
    test["rating_score"] = test["rating"]
    test["fast_score"] = -test["eta_minutes"]
    metrics = {"dataset": "100% sintético, usuários, restaurantes, cliques e horários simulados",
              "seed": seed, "train_sessions": train_end, "validation_sessions": val_end - train_end,
              "test_sessions": sessions - val_end, "candidates_per_session": 12,
              "test": {"model": evaluate(test), "rating_baseline": evaluate(test, "rating_score"),
                       "fastest_baseline": evaluate(test, "fast_score")},
              "validation": validation,
              "limitations": "Métricas offline sintéticas; não estimam conversão, receita, CTR ou resultados de operações reais."}
    ART.mkdir(exist_ok=True)
    joblib.dump(pipeline, ART / "ranker.joblib")
    (ART / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "data").mkdir(exist_ok=True)
    catalog.to_csv(ROOT / "data" / "fictional_restaurants.csv", index=False)
    test.drop(columns=["rating_score", "fast_score"]).head(60).to_csv(ROOT / "data" / "holdout_sample.csv", index=False)
    return metrics

def load_ranker():
    path = ART / "ranker.joblib"
    if not path.exists():
        raise FileNotFoundError("Modelo indisponível. Execute: python scripts/train.py")
    return joblib.load(path)
