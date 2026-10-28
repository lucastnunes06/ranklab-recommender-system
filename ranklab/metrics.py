"""Métricas de ranking por sessão; nenhum feedback online real."""
from __future__ import annotations
import numpy as np
import pandas as pd

def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 5) -> float:
    indices = np.argsort(-np.asarray(scores), kind="stable")[:k]
    labels = np.asarray(labels)
    dcg = float(np.sum((2 ** labels[indices] - 1) / np.log2(np.arange(2, len(indices) + 2))))
    top = np.sort(labels)[::-1][:k]
    idcg = float(np.sum((2 ** top - 1) / np.log2(np.arange(2, len(top) + 2))))
    return dcg / idcg if idcg else 0.0

def evaluate(frame: pd.DataFrame, score_col: str = "score", k: int = 5) -> dict:
    ndcgs, hits, reciprocal, etas, unique, new_ratio = [], [], [], [], [], []
    slices = {}
    for _, session in frame.groupby("session_id", sort=False):
        y = session["clicked"].values
        scores = session[score_col].values
        order = np.argsort(-scores, kind="stable")[:k]
        top = session.iloc[order]
        ndcgs.append(ndcg_at_k(y, scores, k))
        hits.append(int(top["clicked"].any()))
        positives = np.flatnonzero(y[order])
        reciprocal.append(1 / (positives[0] + 1) if len(positives) else 0)
        etas.append(float(top["eta_minutes"].mean()))
        unique.append(float(top["cuisine"].nunique()))
        new_ratio.append(float(top["new_merchant"].mean()))
        pref = str(session["preferred_cuisine"].iloc[0])
        slices.setdefault(pref, []).append(int(top["clicked"].any()))
    return {"ndcg_at_5": round(float(np.mean(ndcgs)), 4),
            "hit_rate_at_5": round(float(np.mean(hits)), 4),
            "mrr_at_5": round(float(np.mean(reciprocal)), 4),
            "mean_eta_top_5": round(float(np.mean(etas)), 2),
            "unique_cuisines_top_5": round(float(np.mean(unique)), 2),
            "new_merchant_share_top_5": round(float(np.mean(new_ratio)), 4),
            "sessions": int(len(hits)),
            "hit_rate_by_preference": {k:round(float(np.mean(v)), 4) for k,v in slices.items()}}
