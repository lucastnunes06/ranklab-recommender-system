# Decisões e experimentos do RankLab

Esta documentação registra **o comportamento implementado**, e não uma história fictícia de desenvolvimento. É um projeto de demonstração com dados simulados e apoio de ferramentas de IA durante a implementação.

## 1. Qual problema o modelo resolve?

Ordenar uma lista de 12 restaurantes candidatos para uma sessão hipotética. A cada sessão variam a cozinha preferida, o orçamento, a pressa e a necessidade de opção vegetariana. A função geradora simula até um clique por sessão; aproximadamente 7% não têm clique.

O algoritmo não aprende a partir de logs reais. As probabilidades são *scores de classificação* condicionados às regras do simulador. Chamar isso de CTR ou efeito de negócio seria incorreto.

## 2. Por que `HistGradientBoostingClassifier`?

Neste protótipo, os atributos são tabelares e o problema foi simplificado para uma classificação por candidato. O algoritmo permite capturar interações não lineares entre, por exemplo, urgência e prazo, sem adicionar um serviço externo ou uma infraestrutura de features.

O custo dessa simplicidade é que o classificador não otimiza diretamente NDCG: a saída é ordenada posteriormente. Uma evolução plausível seria comparar com um ranker pairwise ou listwise nas mesmas partições.

## 3. Por que separar por sessão?

Treino, validação e teste são divididos pelos IDs sequenciais das sessões: 60% / 20% / 20%. Todos os 12 candidatos de uma sessão ficam no mesmo conjunto. O catálogo de 48 restaurantes permanece compartilhado, ou seja, a avaliação não testa estabelecimentos inéditos.

A divisão é adequada para medir a ordenação em sessões futuras *do mesmo simulador*, mas não demonstra generalização para usuários ou mercados reais.

## 4. Baselines e métrica principal

- **NDCG@5:** valoriza itens relevantes nas primeiras cinco posições, permitindo comparar a ordem produzida pelo modelo.
- **Hit Rate@5:** mede se existe pelo menos um clique fictício entre as cinco primeiras opções da sessão.
- **MRR@5:** considera a posição recíproca do primeiro clique dentro das cinco posições.
- **Estrelas e menor ETA:** fornecem referências simples, não modelos concorrentes treinados.

`artifacts/metrics.json` contém os resultados e os cortes de Hit Rate por cozinha preferida. Não os interprete como desempenho de tráfego real.

## 5. Diversidade: uma hipótese que pode piorar a relevância

A função em `ranklab/rerank.py` seleciona cada posição usando:

```text
score_ajustado = score_original - intensidade * 0,042 * número_de_itens_já_escolhidos_da_mesma_cozinha
```

A quantidade `0,042` é uma escolha heurística explícita. A função não altera o score original. Com intensidade zero, a seleção equivale à ordenação descendente do modelo. Com intensidade positiva, restaurantes da mesma categoria recebem uma penalidade crescente nas posições seguintes.

`python scripts/diversity_study.py` testa intensidades fixas `0`, `0,25`, `0,5`, `0,75` e `1`, no mesmo holdout com 290 sessões e 12 candidatos por sessão. A seleção top-5 mede NDCG, Hit Rate, MRR, número de cozinhas distintas, ETA médio e participação de estabelecimentos novos.

O teste mostrou de 3,71 até 4,28 cozinhas distintas em média no top-5, mas NDCG ligeiramente menor no máximo de diversidade (0,5696 → 0,5669). Esses resultados são *descritivos*; nenhum peso foi escolhido por otimização no teste. É possível que mais variedade piore a relevância, e a função de utilidade sintética não captura preferências reais por diversidade.

A API, ao contrário do estudo offline, pode aplicar filtros de tempo e dieta antes do re-ranking, e mostra oito itens por padrão. Não compare diretamente a diversidade das duas situações sem considerar essas diferenças.

## 6. Alterações para tornar o projeto inspecionável

- Um único algoritmo de re-ranking usado pela API e pelo experimento, evitando duas implementações diferentes.
- HTML, CSS e JavaScript separados para facilitar leitura, manutenção e testes.
- Uso de `textContent` para dados recebidos pela API, evitando inserir nomes fictícios diretamente como HTML.
- Proteção contra uma resposta antiga de API sobrescrever uma seleção de filtros mais recente.
- Testes de limites, filtros, estabilidade da ordenação, preservação de scores e métricas.
- Capturas da interface geradas com as respostas reais do backend em `TestClient`, sem fingir conexão com serviços externos.

## 7. Pontos ainda não resolvidos

- **Viés de exposição:** o simulador sorteia cliques independentemente da posição exibida; isso não reflete logs de plataformas reais.
- **Cold start:** os mesmos estabelecimentos podem aparecer no treino e no teste.
- **Calibração:** os scores não foram auditados como probabilidades bem calibradas de cliques reais.
- **Experimentos online:** não há A/B, métricas de receita, satisfação ou impacto operacional.
- **Segurança e operação:** a API local não possui autenticação, limitação de taxa ou monitoramento de produção.
- **Generalização:** os resultados podem refletir apenas as regras da geração de dados.

## 8. Como reproduzir a auditoria

```bash
python scripts/train.py
python scripts/diversity_study.py
pytest -q
```

As figuras do README podem ser recriadas com `python -m docs.capture_dashboard` após instalar as dependências opcionais descritas no README. O GIF documenta cinco estados simulados calculados pelo backend do projeto; não representa uma sessão de uso real.
