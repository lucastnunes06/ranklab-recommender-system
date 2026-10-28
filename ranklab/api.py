"""API local de demonstração do sistema de recomendação RankLab."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .service import get_recommendations, metrics
from .synthetic import CUISINES
from .training import ROOT

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(
    title="RankLab — Recomendação com Machine Learning",
    version="1.2.0",
    description="Demonstração local com sessões e restaurantes sintéticos",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/cuisines")
def cuisines():
    return CUISINES


@app.get("/api/metrics")
def quality():
    return metrics()


def load_study_artifact(filename: str) -> dict:
    """Exibe resultados offline precomputados; não executa o treino na requisição."""
    artifact = ROOT / "artifacts" / filename
    if not artifact.exists():
        raise HTTPException(status_code=503, detail="Execute o script do experimento primeiro")
    return json.loads(artifact.read_text(encoding="utf-8"))


@app.get("/api/robustness-study")
def robustness_study():
    return load_study_artifact("robustness_study.json")


@app.get("/api/cold-start-study")
def cold_start_study():
    return load_study_artifact("cold_start_study.json")


@app.get("/api/diversity-study")
def diversity_study():
    """Resultados do estudo offline, não um ajuste do modelo em tempo real."""
    return load_study_artifact("diversity_study.json")


@app.get("/api/recommendations")
def recommend(
    cuisine: str = "Pizza",
    budget: int = Query(2, ge=1, le=3),
    rush: bool = False,
    diversity: float = Query(0.5, ge=0, le=1),
    vegetarian: bool = False,
    max_eta: int = Query(80, ge=10, le=80),
):
    try:
        return get_recommendations(
            cuisine=cuisine,
            budget=budget,
            rush=rush,
            diversity=diversity,
            vegetarian=vegetarian,
            max_eta=max_eta,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
