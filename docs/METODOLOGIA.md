# Metodologia do RankLab

## Pergunta

Um modelo supervisionado pointwise consegue posicionar um restaurante escolhido em uma sessão **simulada** melhor do que ordenar por avaliação ou prazo de entrega? O que muda se a lista priorizar diversidade de cozinhas?

## Dados

- **48 restaurantes fictícios**, cada um com ID, cozinha, faixa de preço, ETA, nota, taxa, promoção, sinalizador de estabelecimento novo e opção vegetariana.
- **1.450 sessões artificiais**, com 12 candidatos cada, preferências, orçamento, urgência e possível restrição vegetariana.
- Cliques gerados por uma função de utilidade escolhida para o experimento, acrescida de ruído e sessões sem clique.
- Os nomes e parâmetros são fictícios; não há dados pessoais de clientes nem base real de pedidos.

A simulação favorece cozinha preferida, preço compatível e menor ETA quando a sessão tem pressa. Não foi construída para imitar uma distribuição real de mercado.

## Treino e avaliação

- **870 sessões** para treino, **290** para validação e **290** para teste. As sessões são ordenadas pelo ID gerado; não há interseção entre conjuntos de candidatos da mesma sessão.
- `OneHotEncoder` transforma a cozinha favorita e a cozinha do restaurante; atributos numéricos passam adiante.
- `HistGradientBoostingClassifier` estima um score binário por candidato. A lista é ordenada por esse score.
- Comparações não treinadas: nota média decrescente e ETA crescente.
- Métricas calculadas **por sessão e depois agregadas**: NDCG@5, Hit Rate@5 e MRR@5. Sessões sem clique contribuem com zero.
- Informações adicionais: ETA médio, cozinhas distintas no top-5, exposição de estabelecimentos novos e Hit Rate por preferência.

## Experimento de diversidade

O mesmo re-ranker da API (`ranklab/rerank.py`) é aplicado aos scores já calculados no conjunto de teste. Testam-se cinco intensidades previamente fixadas. O relatório é exportado para `artifacts/diversity_study.json`.

É um experimento **descritivo**, não uma busca pelo melhor peso em holdout. Uma escolha de peso para um produto real exigiria separar adequadamente dados para desenvolvimento, teste e decisão de lançamento, além de avaliação online.

## Reprodutibilidade

```bash
python scripts/train.py
python scripts/diversity_study.py
pytest -q
```

Os scripts usam seeds fixas e as mesmas definições de métricas para todas as comparações. O código de avaliação está em `ranklab/metrics.py`.

## Limites

- Métricas offline sintéticas não comprovam conversão, CTR, receita ou preferência de usuários reais.
- O catálogo é compartilhado entre treino e teste. **Não** mede cold start de restaurantes inéditos.
- A simulação não registra propensão de exposição ou viés de posição.
- Diversidade está definida como **quantidade de cozinhas distintas**; não mede diversidade cultural, satisfação nem justiça distributiva.
- Os scores do classificador não são uma função listwise otimizada e não devem ser interpretados como probabilidade real de clique calibrada.
