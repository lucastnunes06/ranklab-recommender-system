"""Validação da penalidade de diversidade + diagnóstico no teste sintético.

O peso é escolhido na VALIDAÇÃO, sem scores do teste. O teste mostra também
as cinco intensidades como diagnóstico exploratório (não para otimização).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ranklab.metrics import evaluate, paired_bootstrap_mean_difference, session_metrics
from ranklab.rerank import DIVERSITY_PENALTY, rerank_indices
from ranklab.synthetic import FEATURES, make_sessions
from ranklab.training import ART, load_ranker, split_sessions

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def score_rankings(frame, level: float):
    scored = frame.copy()
    rank_scores = np.zeros(len(scored), dtype=float)
    for _, group in scored.groupby("session_id", sort=False):
        positions = rerank_indices(
            group["cuisine"].tolist(), group["model_score"].tolist(), diversity=level,
        )
        absolute_positions = scored.index.get_indexer(group.index)
        for order, local_index in enumerate(positions):
            rank_scores[absolute_positions[local_index]] = len(positions) - order
    scored["ranking_position_score"] = rank_scores
    per_session = session_metrics(scored, "ranking_position_score")
    return evaluate(scored, "ranking_position_score"), per_session


def study(seed: int = 17, sessions: int = 1450) -> dict:
    frame, _ = make_sessions(n_sessions=sessions, seed=seed)
    _, validation, holdout = split_sessions(frame, sessions)
    model = load_ranker()
    for part in (validation, holdout):
        part["model_score"] = model.predict_proba(part[FEATURES])[:, 1]

    validation_results = []
    for level in LEVELS:
        result, _ = score_rankings(validation, level)
        validation_results.append({"diversity": level, "ndcg_at_5": result["ndcg_at_5"]})
    # A lista é crescente: em empate, mantemos o menor peso de diversidade.
    selected = max(validation_results, key=lambda row: row["ndcg_at_5"])["diversity"]

    records = []
    reference_ndcg = None
    for level in LEVELS:
        result, by_session = score_rankings(holdout, level)
        ndcg = by_session["ndcg_at_5"].to_numpy()
        if reference_ndcg is None:
            reference_ndcg = ndcg
        delta = paired_bootstrap_mean_difference(ndcg, reference_ndcg, seed=seed + 2026)
        records.append({
            "diversity": level,
            "ndcg_change_vs_zero_ci95": delta,
            **{key: result[key] for key in (
                "ndcg_at_5", "hit_rate_at_5", "mrr_at_5", "unique_cuisines_top_5",
                "mean_eta_top_5", "new_merchant_share_top_5", "sessions",
                "sessions_with_click", "hit_rate_at_5_given_click",
            )},
        })
    chosen_result = next(row for row in records if row["diversity"] == selected)
    payload = {
        "description": "Peso selecionado na validação e avaliado no teste; outras intensidades são diagnósticos exploratórios",
        "seed": seed,
        "total_sessions": sessions,
        "validation_sessions": int(validation["session_id"].nunique()),
        "holdout_sessions": int(holdout["session_id"].nunique()),
        "k": 5,
        "penalty_per_repetition": DIVERSITY_PENALTY,
        "selection": {
            "criterion": "Maior NDCG@5 na validação; desempate a favor da menor intensidade",
            "chosen_diversity": selected,
            "validation_results": validation_results,
            "chosen_holdout_result": chosen_result,
            "holdout_not_used_for_selection": True,
        },
        "levels": records,
        "caveats": (
            "As sessões são artificiais. O teste das demais intensidades é apenas descritivo: "
            "não reescolher o peso olhando esses resultados. "
            "Intervalos bootstrap são por sessão e não equivalem a avaliação online. "
            "A API aplica filtros obrigatórios; este estudo usa todos os 12 candidatos por sessão."
        ),
    }
    ART.mkdir(exist_ok=True)
    (ART / "diversity_study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    report = study()
    print(f"Peso escolhido na validação: {report['selection']['chosen_diversity']}")
    print("Diversidade | NDCG@5 | Cozinhas distintas@5 | Hit Rate@5")
    for row in report["levels"]:
        print(f"{row['diversity']:.2f} | {row['ndcg_at_5']:.4f} | "
              f"{row['unique_cuisines_top_5']:.2f} | {row['hit_rate_at_5']:.4f}")
