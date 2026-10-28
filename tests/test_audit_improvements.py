"""Regressões ligadas às decisões documentadas na auditoria."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ranklab.api import app
from ranklab.metrics import (
    calibration_diagnostics, evaluate, paired_bootstrap_mean_difference,
    session_metrics,
)
from ranklab.service import get_recommendations
from ranklab.synthetic import FEATURES, make_sessions
from ranklab.training import HGB_CANDIDATES, compare, split_sessions
from scripts.cold_start_study import study as cold_start_study

client = TestClient(app)


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_budget_is_a_hard_filter(budget):
    result = get_recommendations("Pizza", budget=budget, k=48)
    assert result["restaurants"]
    assert all(r["price_level"] <= budget for r in result["restaurants"])
    assert all(r["budget_fit"] == 1 for r in result["restaurants"])


def test_combined_filters_and_empty_notice():
    result = get_recommendations("Japonesa", budget=1, vegetarian=True, max_eta=10, k=48)
    assert result["restaurants"] == []
    assert "orçamento" in result["notice"]


def test_invalid_budget_fails_at_api_boundary():
    assert client.get("/api/recommendations?cuisine=Pizza&budget=0").status_code == 422
    with pytest.raises(ValueError):
        get_recommendations("Pizza", budget=4)


def test_synthetic_click_respects_hard_constraints():
    sessions, _ = make_sessions(n_sessions=40, seed=21)
    clicked = sessions[sessions["clicked"] == 1]
    assert (clicked["price_level"] <= clicked["budget_level"]).all()
    assert (clicked["vegetarian_match"] == 1).all()


def test_calibration_is_diagnostic_not_claim_of_real_calibration():
    quality = calibration_diagnostics(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8]), bins=5)
    assert 0 <= quality["brier_score"] <= 1
    assert 0 <= quality["expected_calibration_error"] <= 1
    assert sum(b["count"] for b in quality["reliability_bins"]) == 4
    assert "artificiais" in quality["warning"]
    with pytest.raises(ValueError):
        calibration_diagnostics(np.array([1]), np.array([1.2]))


def test_paired_bootstrap_is_deterministic_and_paired_by_session():
    first = np.array([1.0, .5, 0, 1.0, .4])
    second = np.array([.6, .5, .2, .1, .4])
    a = paired_bootstrap_mean_difference(first, second, repetitions=300)
    b = paired_bootstrap_mean_difference(first, second, repetitions=300)
    assert a == b
    assert a["estimate"] == 0.22
    assert a["ci95"][0] <= a["estimate"] <= a["ci95"][1]
    with pytest.raises(ValueError):
        paired_bootstrap_mean_difference(first, second[:3])


def test_no_click_sessions_reported_explicitly():
    frame = pd.DataFrame({
        "session_id": [1, 1], "clicked": [0, 0], "score": [.7, .3],
        "eta_minutes": [30, 20], "cuisine": ["Pizza", "Árabe"],
        "new_merchant": [0, 1], "preferred_cuisine": ["Pizza", "Pizza"],
    })
    report = evaluate(frame)
    assert report["sessions_without_click"] == 1
    assert report["hit_rate_at_5_given_click"] is None
    assert not session_metrics(frame)["has_click"].any()


def test_compare_selects_hyperparams_using_validation_only():
    result, selected, test, _ = compare(seed=23, sessions=120)
    chosen = result["selection"]
    assert chosen["test_not_used_for_selection"] is True
    assert chosen["selected_parameters"] in HGB_CANDIDATES
    assert chosen["selected_id"] == max(
        chosen["candidates"], key=lambda entry: entry["validation_ndcg_at_5"]
    )["id"]
    assert result["test"]["pairwise_baseline"]["sessions"] == 24
    assert result["test"]["logistic_baseline"]["sessions"] == 24
    assert result["calibration"]["test_synthetic"]["roc_auc"] is not None
    assert len(test) == 24 * 12
    assert len(selected.predict_proba(test[FEATURES])) == 24 * 12


def test_session_ids_do_not_overlap_between_splits():
    frame, _ = make_sessions(n_sessions=25, seed=17)
    parts = split_sessions(frame, sessions=25)
    ids = [set(part["session_id"]) for part in parts]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])


def test_cold_start_holdout_contains_no_leaked_restaurant_ids(tmp_path, monkeypatch):
    from scripts import cold_start_study as module
    monkeypatch.setattr(module, "ART", tmp_path)
    report = cold_start_study(seed=31, sessions=120, holdout_restaurants=6)
    assert not report["training_restaurant_ids_overlap"]
    assert not report["validation_restaurant_ids_overlap"]
    assert report["unseen_candidates"]["candidates"] > 0
    assert (tmp_path / "cold_start_study.json").exists()


def test_diversity_weight_selected_on_validation_and_new_api_routes():
    resp = client.get('/api/diversity-study')
    assert resp.status_code == 200
    selection = resp.json()["selection"]
    assert selection["holdout_not_used_for_selection"] is True
    assert selection["chosen_diversity"] in (0, 0.25, 0.5, 0.75, 1)
    assert selection["chosen_diversity"] == max(
        selection["validation_results"], key=lambda row: row["ndcg_at_5"]
    )["diversity"]
    assert client.get('/api/robustness-study').status_code == 200
    assert client.get('/api/cold-start-study').status_code == 200
    assert client.get('/api/recommendations?cuisine=Pizza').json()["context"]["diversity"] == 0.5


def test_uncalibrated_model_display_is_not_a_click_rate_claim():
    page = client.get('/static/app.js').text
    assert 'Índice do modelo' in page and 'não calibrado' in page
    assert 'Score do modelo:' not in page
    assert 'probabilidades reais calibradas' in client.get('/').text or 'calibradas' in client.get('/').text
