"""Estudo offline: qualidade do ranking x variedade de cozinhas no top-5.

Utiliza o mesmo re-ranker da API. Mantém modelo e conjunto de teste fixos;
não escolhe hiperparâmetros com base nesses resultados.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ranklab.metrics import evaluate
from ranklab.rerank import rerank_indices, DIVERSITY_PENALTY
from ranklab.synthetic import FEATURES, make_sessions
from ranklab.training import ART, load_ranker

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def study(seed: int = 17, sessions: int = 1450) -> dict:
    frame, _ = make_sessions(n_sessions=sessions, seed=seed)
    holdout = frame.loc[frame["session_id"] >= int(0.8 * sessions)].copy()
    holdout["model_score"] = load_ranker().predict_proba(holdout[FEATURES])[:, 1]
    records = []

    for level in LEVELS:
        rank_scores = np.zeros(len(holdout), dtype=float)
        for _, group in holdout.groupby("session_id", sort=False):
            positions = rerank_indices(
                group["cuisine"].tolist(),
                group["model_score"].tolist(),
                diversity=level,
            )
            # Pontuação ordinal apenas para usar o mesmo avaliador de métricas.
            # Estes valores não representam probabilidades do modelo.
            group_positions = holdout.index.get_indexer(group.index)
            for order, local_index in enumerate(positions):
                rank_scores[group_positions[local_index]] = len(positions) - order

        scored = holdout.copy()
        scored["ranking_position_score"] = rank_scores
        result = evaluate(scored, "ranking_position_score")
        records.append({"diversity": level, **{
            key: result[key] for key in (
                "ndcg_at_5", "hit_rate_at_5", "mrr_at_5",
                "unique_cuisines_top_5", "mean_eta_top_5",
                "new_merchant_share_top_5", "sessions",
            )
        }})

    payload = {
        "description": "Estudo de sensibilidade da penalidade de diversidade no holdout sintético",
        "seed": seed,
        "total_sessions": sessions,
        "holdout_sessions": int(holdout["session_id"].nunique()),
        "k": 5,
        "penalty_per_repetition": DIVERSITY_PENALTY,
        "levels": records,
        "caveats": (
            "O catálogo e os cliques são sintéticos; o teste é descritivo. "
            "Este estudo não seleciona o peso ideal nem comprova impacto em usuários reais. "
            "A API aplica filtros antes do re-ranking; este estudo analisa o conjunto de 12 candidatos por sessão."
        ),
    }
    ART.mkdir(exist_ok=True)
    (ART / "diversity_study.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    report = study()
    print("Diversidade | NDCG@5 | Cozinhas distintas@5 | Hit Rate@5")
    for row in report["levels"]:
        print(
            f"{row['diversity']:.2f} | {row['ndcg_at_5']:.4f} | "
            f"{row['unique_cuisines_top_5']:.2f} | {row['hit_rate_at_5']:.4f}"
        )
