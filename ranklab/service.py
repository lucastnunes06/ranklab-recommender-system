"""Recomendações para um catálogo fictício.

O modelo gera os scores; os filtros e o re-ranker decidem quais itens
aparecem e em que ordem. Nenhuma ação externa é executada.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from .rerank import rerank_indices
from .synthetic import CUISINES, FEATURES, candidate_features
from .training import ROOT, load_ranker


@lru_cache(maxsize=1)
def state():
    """Carrega catálogo e modelo uma vez por processo local."""
    catalog = pd.read_csv(ROOT / "data" / "fictional_restaurants.csv")
    model = load_ranker()
    return catalog, model


def reasons_for(restaurant: dict) -> list[str]:
    """Justificativas simples baseadas em atributos, não explicações do ML."""
    reasons = []
    if restaurant["cuisine_match"]:
        reasons.append("cozinha de preferência")
    if restaurant["budget_fit"]:
        reasons.append("dentro do orçamento")
    if restaurant["eta_minutes"] <= 27:
        reasons.append("entrega estimada rápida")
    if restaurant["rating"] >= 4.5:
        reasons.append("avaliação elevada")
    if restaurant["promo"]:
        reasons.append("promoção simulada")
    if restaurant["new_merchant"]:
        reasons.append("estabelecimento novo")
    return reasons[:3] or ["sem correspondências fortes"]


def get_recommendations(
    cuisine: str,
    budget: int = 2,
    rush: bool = False,
    diversity: float = 0.3,
    vegetarian: bool = False,
    max_eta: int = 80,
    k: int = 8,
) -> dict:
    """Aplica filtros, gera scores e reordena sem modificar probabilidades."""
    if cuisine not in CUISINES:
        raise ValueError("Cozinha inválida")

    catalog, model = state()
    frame = candidate_features(catalog, cuisine, budget, rush, vegetarian)
    frame = frame.loc[frame["eta_minutes"] <= max_eta].copy()
    if vegetarian:
        frame = frame.loc[frame["vegetarian_option"] == 1].copy()

    context = {
        "cuisine": cuisine,
        "budget": budget,
        "rush": rush,
        "diversity": diversity,
        "vegetarian": vegetarian,
        "max_eta": max_eta,
    }
    if frame.empty:
        return {
            "restaurants": [],
            "context": context,
            "synthetic": True,
            "notice": "Nenhuma opção atende aos filtros. Amplie o tempo máximo.",
        }

    frame["ml_probability"] = model.predict_proba(frame[FEATURES])[:, 1]
    candidates = frame.to_dict(orient="records")
    ordering = rerank_indices(
        cuisines=[item["cuisine"] for item in candidates],
        scores=[item["ml_probability"] for item in candidates],
        diversity=diversity,
        k=k,
    )

    recommendations = []
    for index in ordering:
        restaurant = candidates[index].copy()
        restaurant["reason"] = reasons_for(restaurant)
        restaurant["ml_probability"] = round(float(restaurant["ml_probability"]), 4)
        recommendations.append(restaurant)

    return {
        "synthetic": True,
        "context": context,
        "restaurants": recommendations,
        "notice": (
            "Scores aprendidos a partir de dados sintéticos; "
            "não representam probabilidades de clique em usuários reais."
        ),
    }


def metrics() -> dict:
    """Retorna o relatório calculado pelo script de treino."""
    return json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
