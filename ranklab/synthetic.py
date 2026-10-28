"""Sessões e restaurantes inteiramente sintéticos, com preferência latente simulada."""
from __future__ import annotations
import numpy as np
import pandas as pd

CUISINES = ["Brasileira", "Japonesa", "Pizza", "Saudável", "Hambúrguer", "Árabe", "Italiana", "Vegana"]
FEATURES = ["cuisine", "preferred_cuisine", "cuisine_match", "budget_level", "price_level", "budget_fit", "rush", "eta_minutes", "rating", "delivery_fee", "promo", "new_merchant", "vegetarian_need", "vegetarian_option", "vegetarian_match"]
CAT = ["cuisine", "preferred_cuisine"]
NUM = [x for x in FEATURES if x not in CAT]

NAMES = [
    'Bistrô Duna', 'Pimenta do Sol', 'Sabor da Vila', 'Estação Ervas',
    'Fornalha Urbana', 'Raiz e Grão', 'Casa do Mirante', 'Tempero Nômade',
    'Ponto do Jardim', 'Oliva & Fogo', 'Quintal da Lua', 'Pomar do Centro',
    'Brasa da Serra', 'Trigo & Mar', 'Canto da Praça', 'Folha de Mel',
    'Aroma Circular', 'Bons Ventos', 'Orvalho Vivo', 'Cozinha Horizonte',
    'Casa Estrela', 'Casa do Lago', 'Flor e Sal', 'Tigela Nativa',
    'Cozinha da Rua', 'Café Jasmim', 'Alho Negro', 'Gengibre Azul',
    'Forno de Pedra', 'Sabores do Norte', 'Prato Alegre', 'Mesa do Bairro',
    'Brisa do Parque', 'Bento Café', 'Doca do Sabor', 'Nuvem & Chá',
    'Ponto Oriental', 'Girassol da Rua', 'Rua da Panela', 'Tâmara Rosa',
    'Oriente Vivo', 'Caju & Canela', 'Beco do Mercado', 'Mesa do Parque',
    'Ritmo do Vale', 'Bambu Bar', 'Horta da Praça', 'Vento Leste',
]

def make_catalog(seed: int = 11, n: int = 48) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n > len(NAMES):
        raise ValueError("n deve ser <= número de nomes fictícios")
    cuisines = rng.choice(CUISINES, n)
    prices = rng.integers(1, 4, n)
    veg = np.array([1 if x in ("Vegana", "Saudável") else int(rng.random() < .36) for x in cuisines])
    return pd.DataFrame({"restaurant_id": [f"REST-{i:03d}" for i in range(1, n + 1)],
                         "name": NAMES[:n], "cuisine": cuisines, "price_level": prices,
                         "base_eta": rng.integers(17, 56, n),
                         "rating": np.round(rng.uniform(3.55, 4.95, n), 2),
                         "delivery_fee": np.round(rng.uniform(0, 12.5, n), 2),
                         "promo": rng.binomial(1, .28, n),
                         "new_merchant": rng.binomial(1, .17, n),
                         "vegetarian_option": veg})

def candidate_features(catalog: pd.DataFrame, pref: str, budget: int, rush: bool, vegetarian_need: bool = False, jitter=None) -> pd.DataFrame:
    df = catalog.copy()
    df["preferred_cuisine"] = pref
    df["cuisine_match"] = (df["cuisine"] == pref).astype(int)
    df["budget_level"] = budget
    df["budget_fit"] = (df["price_level"] <= budget).astype(int)
    df["rush"] = int(rush)
    df["vegetarian_need"] = int(vegetarian_need)
    df["vegetarian_match"] = ((df["vegetarian_option"] == 1) | (not vegetarian_need)).astype(int)
    jitter = np.zeros(len(df)) if jitter is None else jitter
    df["eta_minutes"] = np.clip(df["base_eta"] + jitter, 9, 80).astype(int)
    return df

def make_sessions(n_sessions: int = 1450, n_candidates: int = 12, seed: int = 17):
    rng = np.random.default_rng(seed)
    catalog = make_catalog()
    chunks = []
    for sid in range(n_sessions):
        # Cada sessão tem candidatos que podem competir entre si, sem identificador como feature.
        ids = rng.choice(len(catalog), n_candidates, replace=False)
        pref = str(rng.choice(CUISINES))
        budget = int(rng.integers(1, 4))
        rush = bool(rng.random() < .44)
        veg = bool(rng.random() < .17)
        frame = candidate_features(catalog.iloc[ids], pref, budget, rush, veg, jitter=rng.integers(-6, 8, n_candidates))
        utility = (2.45 * frame["cuisine_match"].values + 1.1 * frame["budget_fit"].values
                   + 0.86 * (frame["rating"].values - 4.0)
                   - 0.036 * frame["eta_minutes"].values * (2.05 if rush else 1.0)
                   - 0.105 * frame["delivery_fee"].values
                   + 0.52 * frame["promo"].values + 0.26 * frame["new_merchant"].values
                   + (1.25 * frame["vegetarian_option"].values if veg else 0)
                   - (2.2 * (frame["vegetarian_option"].values == 0) if veg else 0)
                   + rng.normal(0, .52, n_candidates))
        # Na simulação, orçamento e dieta são RESTRIÇÕES, não só preferências.
        # Sem candidatos elegíveis, a sessão termina sem clique.
        eligible = (frame["price_level"].to_numpy() <= budget)
        if veg:
            eligible &= frame["vegetarian_option"].to_numpy() == 1
        clicked = np.zeros(n_candidates, dtype=int)
        if eligible.any() and rng.random() > .07:
            eligible_positions = np.flatnonzero(eligible)
            values = utility[eligible_positions]
            weights = np.exp(values - values.max())
            weights /= weights.sum()
            clicked[int(rng.choice(eligible_positions, p=weights))] = 1
        frame["session_id"] = sid
        frame["clicked"] = clicked
        chunks.append(frame)
    return pd.concat(chunks, ignore_index=True), catalog
