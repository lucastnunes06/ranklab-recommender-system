"""Sensibilidade do ranking a três seeds simuladas, sem sobrescrever o modelo de demo.

Não equivale a avaliação com dados reais e não seleciona hiperparâmetros pelo teste.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ranklab.training import ART, compare

SEEDS = (17, 29, 41)
SESSIONS = 650


def study(seeds: tuple[int, ...] = SEEDS, sessions: int = SESSIONS) -> dict:
    records = []
    for seed in seeds:
        metrics, _, _, _ = compare(seed=seed, sessions=sessions)
        records.append({
            "seed": seed,
            "selected_model": metrics["selection"]["selected_id"],
            "model_ndcg_at_5": metrics["test"]["model"]["ndcg_at_5"],
            "logistic_ndcg_at_5": metrics["test"]["logistic_baseline"]["ndcg_at_5"],
            "pairwise_ndcg_at_5": metrics["test"]["pairwise_baseline"]["ndcg_at_5"],
            "rating_ndcg_at_5": metrics["test"]["rating_baseline"]["ndcg_at_5"],
        })
    model_scores = np.asarray([r["model_ndcg_at_5"] for r in records])
    report = {
        "description": "Sensibilidade entre seeds: conjuntos independentes do mesmo gerador sintético",
        "seeds": list(seeds), "sessions_per_seed": sessions,
        "results": records,
        "model_ndcg_mean": round(float(model_scores.mean()), 4),
        "model_ndcg_sample_std": round(float(model_scores.std(ddof=1)), 4) if len(records) > 1 else None,
        "caveats": "Não são repetições de produção nem intervalo de confiança externo; somente variedade de sorteios sintéticos.",
    }
    ART.mkdir(exist_ok=True)
    (ART / "robustness_study.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    summary = study()
    for row in summary["results"]:
        print(f"seed={row['seed']} model={row['model_ndcg_at_5']:.4f} "
              f"logistic={row['logistic_ndcg_at_5']:.4f} pairwise={row['pairwise_ndcg_at_5']:.4f}")
