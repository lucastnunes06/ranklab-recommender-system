# Metodologia e limites dos experimentos

## Hipótese

Um modelo de ranking supervisionado consegue ordenar candidatos de um catálogo inventado de forma mais coerente com cliques artificiais do que dois baselines simples? Uma reordenação para evitar cozinhas repetidas pode aumentar a variedade sem piorar substancialmente as métricas de relevância?

Os dados não vêm de entregadores, restaurantes reais ou usuários. A simulação serve para testar código, métricas e escolhas de engenharia — não para representar um mercado.

## Dados e geração

- Catálogo com **48 restaurantes sintéticos**, nome, cozinha, faixas categóricas de preço (`1`, `2`, `3`), nota e ETA.
- **1.450 sessões sintéticas**, cada uma com 12 candidatos e preferências geradas aleatoriamente.
- Os cliques são amostrados a partir de uma função de utilidade criada para o experimento, com ruído. Há sessões sem clique.
- A função de geração **não permite clique** em candidatos acima do limite de preço da sessão ou que violem a restrição vegetariana. Quando não há candidato elegível, a sessão fica sem clique.
- A API também aplica esses filtros **antes do modelo**. O experimento offline, por simplicidade, ordena os 12 candidatos e não mede diretamente o impacto de filtrar o catálogo para cada usuário.

Essa diferença entre avaliação offline e o painel é uma limitação explícita. Para medir desempenho operacional dos filtros, seria necessário avaliar o conjunto elegível de cada sessão separadamente.

## Separação e escolha do modelo

Os IDs de sessão são sequenciais; não contêm timestamps nem simulam uma evolução de distribuição. A divisão de **60% treino / 20% validação / 20% teste** é por ID, não *cronológica* em sentido empírico. Os candidatos de uma mesma sessão não aparecem em conjuntos diferentes, porém **os restaurantes do catálogo são compartilhados**.

O pipeline principal usa OneHotEncoder em duas categorias e `HistGradientBoostingClassifier` nos atributos tabelares. A seleção de `max_iter`, `max_depth` e `learning_rate` utiliza apenas NDCG@5 da **validação** em três configurações pré-definidas. O teste é reservado para avaliação.

Incluí dois benchmarks treinados:

- **Regressão logística pointwise**, útil para verificar se a complexidade extra do boosting faz diferença.
- **Regressão logística pairwise**, treinada sobre diferenças entre vetores de restaurantes clicados e não clicados na mesma sessão. Usa utilidade relativa e não emite probabilidade de clique. Serve somente como comparação offline, não alimenta a API.

Somam-se dois baselines determinísticos, ordenação por nota e menor ETA. **Não** houve busca exaustiva de hiperparâmetros nem treinamento listwise.

O boosting ganhou na validação, mas a regressão logística e o pairwise ficaram com NDCG@5 um pouco superior no teste. Esse fato foi preservado em vez de escolher um modelo pós-teste. Reescolher usando esse resultado contaminaria a comparação futura.

## Definições de métricas

`ranklab/metrics.py` agrupa candidatos por `session_id` antes de computar NDCG@5, Hit Rate@5 e MRR@5. A média é feita sobre sessões, e **sessões sem clique valem zero**. O relatório apresenta separadamente Hit Rate condicional às sessões com clique.

O bootstrap (`paired_bootstrap_mean_difference`) sorteia **sessões inteiras** com reposição, 2.000 vezes por padrão, calculando intervalos percentis de 95% para a diferença de NDCG@5 das mesmas sessões. Isso não resolve vieses de geração, exposição ou generalização.

O diagnóstico probabilístico inclui ROC-AUC, Brier score e ECE de dez faixas para os cliques sintéticos. **ECE baixo não é garantia de calibração em produção**; a saída é rotulada na interface como índice de modelo, não taxa real esperada de clique.

## Diversidade escolhida na validação

A reordenação gulosa penaliza repetições de cozinha:

```text
score_de_ordenacao = score_original − intensidade × 0,042 × ocorrências_anteriores_da_cozinha
```

Os valores 0, 0,25, 0,5, 0,75 e 1 são comparados na **validação**, usando o modelo congelado. O peso `0,5` foi selecionado pelo NDCG@5 da validação e tornou-se o padrão do painel. Resultados dos demais pesos no holdout estão disponíveis somente como *diagnóstico descritivo*, não para nova seleção.

A API aplica antes filtros de preço, dieta e ETA; o estudo considera os doze candidatos originais de cada sessão. Isso limita qualquer comparação direta de diversidade entre resultados da API e métricas offline.

## Estudos exploratórios adicionais

**Robustez de seeds:** refaz o processo com seeds 17, 29 e 41, 650 sessões por execução; os números exibem a variação dos dados gerados, não um intervalo de confiança sobre populações reais.

**Restaurantes inéditos:** seleciona oito IDs sem usar rótulos do teste, exclui todos os registros desses IDs no treino e na validação, seleciona configuração na validação restante e mede ROC-AUC/Brier por candidato visto ou não visto no teste. Isto **não** mede ranking cold-start de usuários e não remove atributos conhecidos de um estabelecimento.

Esses scripts geram apenas relatórios; o arquivo binário principal usado pelo painel não é modificado por eles.

## Reproduzir

```bash
python scripts/train.py
python scripts/diversity_study.py
python scripts/robustness_study.py
python scripts/cold_start_study.py
pip install -r requirements-dev.txt
ruff check . --select E4,E7,E9,F
pytest --cov=ranklab --cov-report=term-missing --cov-fail-under=90
```

Os artefatos resultantes ficam em `artifacts/`. Os arquivos `.joblib` são binários que só devem ser carregados de origem confiável.

## Limites que permanecem

A distribuição artificial favorece atributos escolhidos no gerador. O projeto não mede cliques verdadeiros, propensão de exposição, vieses por posição, testes A/B, satisfação real, ranking de catálogo novo em escala ou usuários sem histórico. A API local não implementa autenticação, observabilidade, rate limiting, persistência ou controles de produção. Não há recomendação personalizada a partir de comportamento individual histórico.
