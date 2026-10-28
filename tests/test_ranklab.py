import numpy as np
import pytest
from fastapi.testclient import TestClient
from ranklab.api import app
from ranklab.metrics import ndcg_at_k, evaluate
from ranklab.rerank import rerank_indices
from ranklab.synthetic import make_catalog, make_sessions
from ranklab.service import get_recommendations

client = TestClient(app)

def test_ndcg_perfect():
    assert abs(ndcg_at_k(np.array([1,0,0]), np.array([.9,.1,.2]), k=3) - 1) < 1e-9

def test_ndcg_worst():
    assert ndcg_at_k(np.array([1,0,0]), np.array([.1,.9,.8]), k=3) < 1

def test_generator_reproducible():
    a,b = make_sessions(6, 12, 31)
    x,y = make_sessions(6, 12, 31)
    assert a.equals(x) and b.equals(y)

def test_no_real_data():
    assert all(name not in ['Rede Real A', 'Rede Real B'] for name in make_catalog().name)

def test_endpoint_and_budget():
    r = client.get('/api/recommendations?cuisine=Pizza&budget=2&diversity=0.6')
    assert r.status_code == 200
    data = r.json()
    assert data['synthetic'] and len(data['restaurants']) == 8
    assert all(0 <= x['ml_probability'] <= 1 for x in data['restaurants'])

def test_max_eta_and_veg_filter():
    r = client.get('/api/recommendations?cuisine=Vegana&vegetarian=true&max_eta=40')
    assert r.status_code == 200
    for item in r.json()['restaurants']:
        assert item['eta_minutes'] <= 40 and item['vegetarian_option'] == 1

def test_bad_cuisine():
    assert client.get('/api/recommendations?cuisine=inexistente').status_code == 422

def test_metrics_api():
    resp = client.get('/api/metrics')
    assert resp.status_code == 200 and 'ndcg_at_5' in resp.json()['test']['model']

# Os testes abaixo verificam comportamento, não só a presença dos módulos.

def test_rerank_zero_diversity_is_score_order():
    indices = rerank_indices(['Pizza', 'Pizza', 'Árabe'], [.9, .89, .88], 0)
    assert indices == [0, 1, 2]


def test_diversity_can_change_order_not_score():
    scores = [.9, .89, .88]
    assert rerank_indices(['Pizza', 'Pizza', 'Árabe'], scores, 1) == [0, 2, 1]
    assert scores == [.9, .89, .88]


def test_rerank_stable_ties_and_limits():
    assert rerank_indices(['Pizza', 'Árabe', 'Italiana'], [.5, .5, .5], 0, k=2) == [0, 1]
    assert rerank_indices([], [], 0) == []
    assert rerank_indices(['Pizza'], [.5], 0, k=0) == []


@pytest.mark.parametrize('level', [-0.01, 1.01])
def test_rerank_rejects_invalid_weights(level):
    with pytest.raises(ValueError):
        rerank_indices(['Pizza'], [.7], level)


def test_rerank_rejects_invalid_input_shapes():
    with pytest.raises(ValueError):
        rerank_indices(['Pizza'], [.7, .8], 0)
    with pytest.raises(ValueError):
        rerank_indices(['Pizza'], [.7], 0, k=-1)


def test_same_probabilities_regardless_of_diversity():
    low = get_recommendations('Pizza', diversity=0, k=48)
    high = get_recommendations('Pizza', diversity=1, k=48)
    low_map = {x['restaurant_id']: x['ml_probability'] for x in low['restaurants']}
    high_map = {x['restaurant_id']: x['ml_probability'] for x in high['restaurants']}
    assert low_map == high_map


def test_mobile_frontend_assets_are_served():
    response = client.get('/')
    assert response.status_code == 200
    assert 'static/styles.css' in response.text and 'static/app.js' in response.text
    assert client.get('/static/styles.css').status_code == 200
    assert client.get('/static/app.js').status_code == 200


def test_study_endpoint_has_reproducible_levels():
    response = client.get('/api/diversity-study')
    assert response.status_code == 200
    payload = response.json()
    assert len(payload['levels']) == 5
    assert payload['holdout_sessions'] == 290
    assert all(0 <= row['ndcg_at_5'] <= 1 for row in payload['levels'])
    assert payload['levels'][0]['ndcg_at_5'] == client.get('/api/metrics').json()['test']['model']['ndcg_at_5']


def test_no_click_session_has_zero_ranking_metrics():
    import pandas as pd
    frame = pd.DataFrame({
        'session_id':[1, 1], 'clicked':[0, 0], 'score':[.8, .3],
        'eta_minutes':[18, 32], 'cuisine':['Pizza', 'Árabe'], 'new_merchant':[0,1],
        'preferred_cuisine':['Pizza','Pizza'],
    })
    result = evaluate(frame)
    assert result['ndcg_at_5'] == 0
    assert result['hit_rate_at_5'] == 0
    assert result['mrr_at_5'] == 0


def test_sessions_are_separate_and_have_at_most_one_click():
    frame, _ = make_sessions(18, 12, 17)
    assert (frame.groupby('session_id').size() == 12).all()
    assert (frame.groupby('session_id')['clicked'].sum() <= 1).all()


def test_no_real_customers_in_demo_catalog():
    frame = make_catalog()
    assert frame['restaurant_id'].is_unique
    assert set(frame['restaurant_id'].str[:5]) == {'REST-'}
    assert all(not name.isdigit() for name in frame['name'])
