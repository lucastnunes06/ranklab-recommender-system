"""Comparador pairwise linear para estudo offline, não é usado pela API.

Cada clique gera pares positivo-negativo na mesma sessão. Treina ambos os
sentidos do par para que a fronteira não dependa da ordem de entrada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .synthetic import CAT, FEATURES, NUM


@dataclass
class PairwiseRanker:
    preprocessor: ColumnTransformer
    classifier: LogisticRegression

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.preprocessor.transform(frame[FEATURES])
        # Um valor de utilidade relativo; não é probabilidade de clique.
        return np.asarray(features @ self.classifier.coef_[0], dtype=float).ravel()


def fit_pairwise(frame: pd.DataFrame, seed: int = 17) -> PairwiseRanker:
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", StandardScaler(), NUM),
    ])
    transformed = preprocessor.fit_transform(frame[FEATURES])
    diffs = []
    for positions in frame.groupby("session_id", sort=False).indices.values():
        selected = positions[frame["clicked"].to_numpy()[positions] == 1]
        if not len(selected):
            continue
        negatives = positions[frame["clicked"].to_numpy()[positions] == 0]
        diffs.append(transformed[selected[0]] - transformed[negatives])
    if not diffs:
        raise ValueError("O treino pairwise precisa de sessões com pelo menos um clique")
    positive = np.concatenate(diffs)
    pairs = np.vstack((positive, -positive))
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(positive))))
    classifier = LogisticRegression(max_iter=600, random_state=seed, fit_intercept=False)
    classifier.fit(pairs, labels)
    return PairwiseRanker(preprocessor, classifier)
