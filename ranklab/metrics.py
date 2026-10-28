"""Ranking, inferência e incerteza: métricas agregadas por sessão.

Não há dados de usuários reais nem propensão de exposição neste simulador.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    indices = np.argsort(-np.asarray(scores), kind="stable")[:k]
    labels = np.asarray(labels)
    dcg = float(np.sum((2 ** labels[indices] - 1) / np.log2(np.arange(2, len(indices) + 2))))
    top = np.sort(labels)[::-1][:k]
    idcg = float(np.sum((2 ** top - 1) / np.log2(np.arange(2, len(top) + 2))))
    return dcg / idcg if idcg else 0.0


def session_metrics(frame: pd.DataFrame, score_col: str = "score", k: int = 5) -> pd.DataFrame:
    """Mantém sessões sem clique, identificadas explicitamente por has_click."""
    rows = []
    for sid, session in frame.groupby("session_id", sort=False):
        y = session["clicked"].to_numpy()
        scores = session[score_col].to_numpy()
        order = np.argsort(-scores, kind="stable")[:k]
        top = session.iloc[order]
        positives = np.flatnonzero(y[order])
        rows.append({
            "session_id": sid,
            "preferred_cuisine": str(session["preferred_cuisine"].iloc[0]),
            "has_click": bool(y.any()),
            "ndcg_at_5": ndcg_at_k(y, scores, k),
            "hit_rate_at_5": float(top["clicked"].any()),
            "mrr_at_5": float(1 / (positives[0] + 1)) if len(positives) else 0.0,
            "mean_eta_top_5": float(top["eta_minutes"].mean()),
            "unique_cuisines_top_5": float(top["cuisine"].nunique()),
            "new_merchant_share_top_5": float(top["new_merchant"].mean()),
        })
    return pd.DataFrame(rows)


def evaluate(frame: pd.DataFrame, score_col: str = "score", k: int = 5) -> dict:
    per_session = session_metrics(frame, score_col, k)
    if per_session.empty:
        raise ValueError("Não há sessões para avaliação")
    report = {
        column: round(float(per_session[column].mean()), 4 if column in
                      ("ndcg_at_5", "hit_rate_at_5", "mrr_at_5", "new_merchant_share_top_5") else 2)
        for column in (
            "ndcg_at_5", "hit_rate_at_5", "mrr_at_5", "mean_eta_top_5",
            "unique_cuisines_top_5", "new_merchant_share_top_5",
        )
    }
    clicked = per_session.loc[per_session["has_click"]]
    report.update({
        "sessions": int(len(per_session)),
        "sessions_with_click": int(len(clicked)),
        "sessions_without_click": int(len(per_session) - len(clicked)),
        "hit_rate_at_5_given_click": round(float(clicked["hit_rate_at_5"].mean()), 4) if len(clicked) else None,
        "hit_rate_by_preference": {
            str(cuisine): round(float(group["hit_rate_at_5"].mean()), 4)
            for cuisine, group in per_session.groupby("preferred_cuisine")
        },
    })
    return report


def calibration_diagnostics(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict:
    """Diagnóstico no simulador. ECE baixo não implica calibração no mundo real."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.size == 0 or p.size != y.size or np.any((p < 0) | (p > 1)):
        raise ValueError("Rótulos e scores precisam ser não vazios, alinhados e no intervalo [0,1]")
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for idx in range(bins):
        mask = bucket == idx
        if not mask.any():
            continue
        confidence, frequency = float(p[mask].mean()), float(y[mask].mean())
        ece += mask.mean() * abs(confidence - frequency)
        reliability.append({
            "range": [round(idx / bins, 2), round((idx + 1) / bins, 2)],
            "count": int(mask.sum()),
            "mean_score": round(confidence, 4),
            "observed_click_rate": round(frequency, 4),
        })
    return {
        "brier_score": round(float(brier_score_loss(y, p)), 4),
        "expected_calibration_error": round(float(ece), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) == 2 else None,
        "simulated_click_rate": round(float(y.mean()), 4),
        "mean_model_score": round(float(p.mean()), 4),
        "reliability_bins": reliability,
        "warning": "Calibração medida em cliques artificiais: não é probabilidade confiável de clique real.",
    }


def paired_bootstrap_mean_difference(
    reference: np.ndarray,
    alternative: np.ndarray,
    seed: int = 2026,
    repetitions: int = 2000,
) -> dict:
    """Intervalo percentil para diferença média pareada (reference - alternative).

    Reamostra SESSÕES, não candidatos individuais. Não é uma avaliação online.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(alternative, dtype=float)
    if not a.size or a.shape != b.shape or a.ndim != 1 or repetitions < 100:
        raise ValueError("Séries pareadas inválidas ou número insuficiente de replicações")
    rng = np.random.default_rng(seed)
    differences = a - b
    bootstrap = differences[rng.integers(0, a.size, size=(repetitions, a.size))].mean(axis=1)
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "estimate": round(float(differences.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "sessions": int(a.size),
        "replications": repetitions,
        "seed": seed,
    }
