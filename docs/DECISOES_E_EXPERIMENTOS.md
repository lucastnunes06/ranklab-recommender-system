# Decisões técnicas do RankLab

A intenção deste documento é registrar **o estado efetivo do código**, os compromissos assumidos e resultados reproduzidos. Não descreve uma história fictícia de desenvolvimento manual. O projeto foi preparado com assistência de IA, e o material é disponibilizado para revisão.

## 1. Preço: por que é filtro obrigatório?

O comportamento inicial permitia recomendar restaurante fora da faixa selecionada, porque `budget_fit` era somente uma feature do modelo. Isso não respeitava a interpretação da interface.

Agora o serviço executa `price_level <= budget` **antes** da pontuação. O gerador de cliques também restringe a escolha a restaurantes elegíveis em preço e dieta. O preço é **categórico**, não um valor monetário exato. Os testes verificam os três níveis e combinações sem resultados.

**Trade-off:** o conjunto de itens disponíveis diminui; em cenários extremos pode ser vazio. A mensagem da API apresenta esse caso sem afrouxar um filtro silenciosamente.

## 2. Scores não são promessas de probabilidade

A API mantém o campo legado `ml_probability` para compatibilidade, mas sua saída representa uma confiança de classificador treinado em simulação. A interface utiliza `Índice do modelo: 0.xxx (não calibrado)` e a resposta inclui `score_interpretation`.

Implementamos Brier score, ECE por faixas e ROC-AUC em holdout sintético. Mesmo com ECE de `0,0093`, isso não autoriza uma afirmação de probabilidade de clique real: os rótulos foram gerados por regras escritas no próprio projeto.

## 3. Validação por sessão, não por cronologia real

Os IDs ordenados impedem vazamento de candidatos entre treino, validação e teste da mesma sessão. Entretanto, a geração não representa mudança no tempo. Documentamos a divisão como **sequencial por sessão**. O catálogo é compartilhado entre conjuntos.

Três configurações HGB são comparadas por NDCG@5 na validação. A escolhida foi a terceira (125 iterações, profundidade 4, learning rate 0,08), aplicada ao teste sem reescolha. Foram incluídos baselines supervisionados logístico e pairwise.

No teste, logística (`0,6710`) e pairwise (`0,6714`) superaram o boosting (`0,6559`) em NDCG@5. O bootstrap pareado de diferença **boosting menos pairwise** gerou IC95% `[-0,0356; 0,0051]`; inclui zero. O resultado correto não é declarar o boosting como vencedor absoluto, nem migrar para pairwise usando o teste como critério.

## 4. Diversidade: seleção na validação

O re-ranker recebe a lista de scores e penaliza a repetição de cozinha. Intensidade `0,5` foi escolhida pela validação antes da leitura do teste e virou o padrão da API. Outros pesos no teste são mostrados somente como sensibilidade exploratória. Em comparação com intensidade zero, `0,5` levou o NDCG@5 de `0,6559` a `0,6637` e a variedade de `3,81` para `4,18` cozinhas por top-5, sempre no gerador sintético.

A função de penalização de `0,042` é uma escolha de engenharia, não otimização de produto. Nenhuma mudança no score original acontece durante o re-ranking.

## 5. Experimentos adicionais

- **Seeds diferentes:** NDCG@5 do boosting entre `0,6198` e `0,6680`, ao mudar dados aleatórios do mesmo gerador. A configuração vencedora na validação varia entre as seeds. A variabilidade existe mesmo sem alterações no código.
- **Restaurantes inéditos:** oito IDs são retirados do treino e da validação em estudo separado. Os resultados por candidato são `0,8796` (vistos) e `0,8873` (não vistos) em ROC-AUC, com atributos disponíveis em ambos os grupos. **Não** é uma simulação completa de cold start.
- **Pairwise:** comparação com um algoritmo treinado em pares positivo/negativo da mesma sessão, sem uso no endpoint de recomendação.

O projeto ainda não corrige viés de posição, não dispõe de propensões de exposição nem compara perda listwise ou resultados de usuários reais. Essas são limitações assumidas, não features escondidas.

## 6. Qualidade de engenharia e demonstração

O código separa gerador, modelos, métricas, regras de elegibilidade, re-ranking e API. A captura do painel é feita com os arquivos de HTML/CSS/JS efetivos e respostas calculadas pelo backend local em `TestClient`. Não há imagem gerada para simular um produto inexistente.

Os testes de integração incluem filtros rígidos, rotas, ausência de cliques, seleção de pesos, bootstrap determinístico e exclusão de restaurantes inéditos do treino. A suíte contém **34 testes**, cobertura observada de **95%** localmente e mínimo de 90% exigido no CI, além de lint de erros Python e auditoria consultiva das bibliotecas.

O histórico de commits não foi manipulado para criar uma narrativa artificial. Melhorias futuras devem ser registradas em mudanças reais com hipóteses, resultados e eventuais problemas encontrados.

## 7. Execução dos experimentos

```bash
python scripts/train.py
python scripts/diversity_study.py
python scripts/robustness_study.py
python scripts/cold_start_study.py
pytest -q
```

O resultado do boosting no conjunto sintético é consultável em [`../artifacts/metrics.json`](../artifacts/metrics.json); o estudo de diversidade em [`../artifacts/diversity_study.json`](../artifacts/diversity_study.json). Não há validação em produção, APIs de restaurantes reais ou tratamento de dados pessoais.
