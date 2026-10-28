"""Avaliação exploratória de restaurantes cujo ID não apareceu no treino.

Compara scores em restaurantes vistos e não vistos. Como o catálogo tem
features disponíveis, isto não mede cold start de usuários ou entidades sem
metadados; são métricas por candidato e não ranking por sessão.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from ranklab.synthetic import FEATURES, make_sessions
from ranklab.training import ART, HGB_CANDIDATES, make_hgb, split_sessions


def _evaluate_subset(model, data):
    y = data["clicked"].to_numpy()
    p = model.predict_proba(data[FEATURES])[:, 1]
    return {
        "candidates": int(len(data)),
        "positives": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) == 2 else None,
        "brier_score": round(float(brier_score_loss(y, p)), 4),
    }


def study(seed: int = 17, sessions: int = 1450, holdout_restaurants: int = 8):
    frame, catalog = make_sessions(n_sessions=sessions, seed=seed)
    train, val, test = split_sessions(frame, sessions)
    rng = np.random.default_rng(seed + 913)
    held_out = sorted(rng.choice(catalog["restaurant_id"], holdout_restaurants, replace=False).tolist())
    clean_train = train.loc[~train["restaurant_id"].isin(held_out)].copy()
    clean_val = val.loc[~val["restaurant_id"].isin(held_out)].copy()
    val_scores = []
    for idx, params in enumerate(HGB_CANDIDATES):
        model = make_hgb(params, seed)
        model.fit(clean_train[FEATURES], clean_train["clicked"])
        pred = model.predict_proba(clean_val[FEATURES])[:, 1]
        # Comparação por candidato em validação, sem rótulos do teste.
        auc = roc_auc_score(clean_val["clicked"], pred)
        val_scores.append((float(auc), -idx, model, params))
    _, _, best, params = max(val_scores, key=lambda x: (x[0], x[1]))
    unseen = test.loc[test["restaurant_id"].isin(held_out)]
    seen = test.loc[~test["restaurant_id"].isin(held_out)]
    report = {
        "description": "Atributos de 8 restaurantes não observados no treino/validação; avaliação apenas por candidato",
        "seed": seed,
        "held_out_restaurant_ids": held_out,
        "training_restaurant_ids_overlap": bool(clean_train["restaurant_id"].isin(held_out).any()),
        "validation_restaurant_ids_overlap": bool(clean_val["restaurant_id"].isin(held_out).any()),
        "chosen_params_using_validation": params,
        "seen_candidates": _evaluate_subset(best, seen),
        "unseen_candidates": _evaluate_subset(best, unseen),
        "limitations": (
            "Mede score em metadados de restaurantes inéditos, não qualidade de ranking em cold start de usuários. "
            "Os rótulos são criados pelo mesmo gerador, portanto não há validação externa."
        ),
    }
    ART.mkdir(exist_ok=True)
    (ART / "cold_start_study.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = study()
    print(json.dumps(result, ensure_ascii=False, indent=2))
