"""Reordenação gulosa explícita, compartilhada entre a API e os experimentos.

O score do modelo permanece inalterado. A penalidade é aplicada somente para
escolher a próxima posição da lista. Não se trata de um ranking treinado online.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

DIVERSITY_PENALTY = 0.042  # hipótese de projeto, não valor estimado com usuários reais


def rerank_indices(
    cuisines: Sequence[str],
    scores: Sequence[float],
    diversity: float = 0.3,
    k: int | None = None,
) -> list[int]:
    """Retorna índices posicionais, mantendo estabilidade em caso de empate.

    Para cada posição, subtrai do score original uma penalidade por ocorrência
    anterior da mesma cozinha. A lista de scores recebida não é modificada.
    """
    if not 0 <= diversity <= 1:
        raise ValueError("diversity deve estar entre 0 e 1")
    if len(cuisines) != len(scores):
        raise ValueError("Cozinhas e scores devem ter o mesmo comprimento")
    if k is not None and k < 0:
        raise ValueError("k não pode ser negativo")

    pending = list(range(len(scores)))
    selected: list[int] = []
    counts: Counter[str] = Counter()
    limit = len(pending) if k is None else min(k, len(pending))

    while pending and len(selected) < limit:
        # Em empates, escolhe o índice original menor (resultado reprodutível).
        best = max(
            pending,
            key=lambda i: (
                float(scores[i])
                - diversity * DIVERSITY_PENALTY * counts[cuisines[i]],
                -i,
            ),
        )
        pending.remove(best)
        selected.append(best)
        counts[cuisines[best]] += 1

    return selected
