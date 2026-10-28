# RankLab

**O que vale mais em uma recomendação: relevância, rapidez, preço ou variedade?**

[![Testes](https://github.com/lucastnunes06/ranklab-recommender-system/actions/workflows/ci.yml/badge.svg)](https://github.com/lucastnunes06/ranklab-recommender-system/actions)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Licença](https://img.shields.io/badge/Licen%C3%A7a-MIT-607D8B)

O RankLab é um laboratório local de recomendações de restaurantes. O problema é simples de explicar, mas tem escolhas que não são tão simples: um restaurante bem avaliado pode ser caro, a opção mais rápida pode não combinar com o gosto da pessoa e uma lista inteira da mesma cozinha pode ser pouco útil.

O projeto usa **48 restaurantes fictícios e 1.450 sessões geradas artificialmente** para testar essas decisões. O objetivo é exercitar engenharia e avaliação de ranking — **não** representar um serviço real ou prever preferências de pessoas reais.

## Teste o painel

![Painel do RankLab renderizado no navegador com respostas do backend local](docs/dashboard.png)

[Ver a demonstração em GIF](docs/recommendations-demo.gif) · [Código da captura](docs/capture_dashboard.py)

O painel permite escolher cozinha, **limite de preço obrigatório** (`$`, `$$` ou `$$$`), tempo máximo, necessidade de opção vegetariana e intensidade de diversidade. O backend elimina opções incompatíveis **antes** de aplicar o modelo. Se não houver restaurantes elegíveis, a interface informa que os filtros precisam ser ajustados.

O número exibido em cada cartão é o **índice do classificador** no conjunto sintético, não uma previsão calibrada de clique real. Os motivos textuais vêm de atributos dos restaurantes e não pretendem explicar causalmente a decisão do ML.

## O que está implementado

| Camada | Implementação |
| --- | --- |
| Dados | Gerador reproduzível com restrições de preço e dieta, sessões com até um clique e casos sem clique |
| Modelo principal | Gradient boosting pointwise com **três configurações comparadas na validação** |
| Comparações | Regressão logística, ranker **pairwise linear**, estrelas e menor tempo estimado |
| Avaliação | NDCG@5, Hit Rate@5, MRR@5, bootstrap pareado por sessão e diagnóstico de calibração |
| Re-ranking | Penalidade para cozinhas repetidas; peso escolhido na **validação**, sem consulta ao teste |
| Estudos adicionais | Três seeds independentes do mesmo gerador e teste exploratório com IDs de restaurantes inéditos |
| Aplicação | API FastAPI + HTML, CSS e JavaScript separados; endpoints e testes funcionais |
| Qualidade | `pytest`, cobertura, lint no CI e verificação de dependências em caráter consultivo |

A recomendação usa um **classificador pointwise**, e a comparação pairwise é um benchmark offline. Ainda não existe treinamento listwise, plataforma de eventos ou banco de dados em produção.

## Instalação e reprodução

Requisitos: Python 3.11+, Git. Não são necessárias credenciais ou serviços externos.

```bash
git clone https://github.com/lucastnunes06/ranklab-recommender-system.git
cd ranklab-recommender-system
python -m venv .venv
```

Linux/macOS: `source .venv/bin/activate`  
Windows PowerShell: `.venv\Scripts\Activate.ps1`

```bash
pip install -r requirements.txt
python scripts/train.py
python scripts/diversity_study.py
python scripts/robustness_study.py
python scripts/cold_start_study.py
uvicorn ranklab.api:app --reload --port 8001
```

Abra **http://127.0.0.1:8001** para usar o painel e **http://127.0.0.1:8001/docs** para explorar os endpoints.

```bash
curl 'http://127.0.0.1:8001/api/recommendations?cuisine=Pizza&budget=1&rush=true&diversity=0.5&max_eta=45'
```

Nesse exemplo, restaurantes nas faixas `2` e `3` **não aparecem**, mesmo que tenham score alto. As faixas não equivalem a preços exatos em reais. A aplicação também filtra opção vegetariana e tempo máximo quando solicitados.

Os arquivos em `artifacts/` são destinados **somente à demonstração**. Nunca carregue arquivos `.joblib`/pickle desconhecidos: o formato executa código ao ser desserializado.

## Experimento 1 — escolher modelo sem olhar o teste

As sessões foram divididas **por ID sequencial**, com todos os candidatos de cada sessão no mesmo conjunto: **870 para treino, 290 para validação e 290 para teste**. Não existem timestamps de eventos reais; não chamamos essa divisão de validação temporal.

Três configurações de gradient boosting foram treinadas e comparadas pelo **NDCG@5 na validação**. A vencedora (`125` iterações, profundidade `4`, learning rate `0,08`) foi então avaliada uma única vez no teste, junto com os baselines.

| Estratégia | NDCG@5 no teste | Hit Rate@5 | MRR@5 |
| --- | ---: | ---: | ---: |
| **Gradient boosting escolhido na validação** | 0,6559 | 0,8448 | 0,5926 |
| **Regressão logística** | **0,6710** | **0,8655** | 0,6058 |
| **Pairwise linear** | **0,6714** | 0,8621 | **0,6072** |
| Estrelas | 0,3346 | 0,5552 | 0,2630 |
| Menor tempo estimado | 0,3932 | 0,6207 | 0,3191 |

**Um resultado que vale discutir:** a configuração de gradient boosting ganhou na validação, mas os modelos lineares tiveram NDCG maior no teste. Isso **não** justifica substituí-la com base nesses números; fazer isso usaria o conjunto de teste para escolher o vencedor. Os intervalos bootstrap de 95% para a diferença entre boosting e logística/pairwise incluem zero. A comparação sugere que um modelo mais complexo não é automaticamente melhor.

O Hit Rate@5 acima inclui **18 sessões sem clique** (contabilizadas como zero). Entre as **272 sessões com clique**, o Hit Rate@5 do boosting foi **0,9007**. Os dois denominadores são documentados no JSON.

[Relatório completo, seleção e intervalos](artifacts/metrics.json) · [Código de treino](ranklab/training.py)

## Experimento 2 — variedade sem escolher o peso no teste

Uma penalidade explícita diminui a prioridade de cozinhas que já apareceram na lista. O algoritmo preserva o score original do modelo. Os pesos `0`, `0,25`, `0,50`, `0,75` e `1` foram comparados **na validação**, e `0,50` foi selecionado por NDCG@5, com desempate a favor do menor peso.

Abaixo estão os resultados no holdout, incluindo as outras intensidades apenas para análise descritiva:

| Diversidade | NDCG@5 | Cozinhas diferentes no top 5 | Hit Rate@5 |
| --- | ---: | ---: | ---: |
| 0% | 0,6559 | 3,81 | 0,8448 |
| 25% | 0,6609 | 4,10 | 0,8552 |
| **50% (escolhido na validação)** | **0,6637** | **4,18** | **0,8621** |
| 75% | 0,6628 | 4,24 | 0,8586 |
| 100% | 0,6623 | 4,30 | 0,8586 |

A diferença de NDCG entre 50% e 0% foi **0,0078**, com intervalo percentil bootstrap pareado de 95% de **[0,0020; 0,0146]**. É um resultado do **simulador**, sujeito à definição de rótulos e à pequena quantidade de sessões, não evidência de efeito em pessoas.

[Estudo e incerteza por peso](artifacts/diversity_study.json) · [Implementação compartilhada do re-ranking](ranklab/rerank.py)

## Experimento 3 — calibração, seeds e cold start

**Calibração em dados sintéticos.** O classificador teve Brier score `0,0585`, ECE com 10 faixas `0,0093` e ROC-AUC `0,8826`. Esses valores avaliam a correspondência com os rótulos artificiais usados pelo simulador; **não** demonstram probabilidades de clique real confiáveis. O painel usa o termo *índice* para evitar essa interpretação. [Curva de confiabilidade por faixas (JSON)](artifacts/metrics.json).

**Variação entre seeds.** Rodando o pipeline com três seeds e 650 sessões por execução, o NDCG@5 do modelo variou de `0,6198` a `0,6680` (média `0,6372`). São três sorteios do mesmo gerador; isso mostra sensibilidade interna, não generalização para uma população real. [Dados](artifacts/robustness_study.json).

**Restaurantes não vistos no treino.** Um estudo à parte reserva oito IDs de restaurante, exclui esses registros do treino e da validação, e mede ROC-AUC por candidato no teste: `0,8796` para restaurantes vistos versus `0,8873` para os não vistos. Aqui ainda existem atributos completos, e **não** se mede ranking cold-start de usuários, nem estabelecimentos sem metadados. [Dados](artifacts/cold_start_study.json).

## Qualidade e testes

```bash
pip install -r requirements-dev.txt
ruff check . --select E4,E7,E9,F
pytest -q --cov=ranklab --cov-report=term-missing --cov-fail-under=90
```

A suíte foi ampliada para **34 testes**, incluindo restrições de preço, dieta, casos sem resultados, integridade dos splits, bootstrap, estudo com restaurantes inéditos e comportamento da API. Na execução local, a cobertura do pacote Python atingiu **95%**; o CI repete as avaliações e exige pelo menos 90% de cobertura. A checagem de vulnerabilidades das dependências no CI é **consultiva** e não substitui revisão antes de um deploy.

Rotas úteis: `GET /api/cuisines`, `GET /api/recommendations`, `GET /api/metrics`, `GET /api/diversity-study`, `GET /api/robustness-study`, `GET /api/cold-start-study`.

Para gerar a imagem e o GIF com a **interface verdadeira e respostas do backend local**:

```bash
pip install -r requirements-preview.txt
python -m playwright install chromium
python -m docs.capture_dashboard
```

## Organização e decisões

```text
ranklab/
  synthetic.py       gerador de catálogo e sessões sintéticas
  training.py        experimentos, escolha em validação e baselines
  pairwise.py        ranker linear par-a-par para estudo offline
  metrics.py         métricas por sessão, calibração e bootstrap
  rerank.py          diversidade com penalidade transparente
  service.py         filtros obrigatórios e recomendações
  api.py             API local
  static/            HTML, CSS e JavaScript separados
scripts/             treino e experimentos reproduzíveis
tests/               testes unitários e de integração
artifacts/           métricas e modelo local gerados
docs/                decisões, metodologia e demonstrações
.github/workflows/   CI e auditoria consultiva de dependências
```

Detalhes do método, alternativas consideradas e limites: [Metodologia](docs/METODOLOGIA.md) · [Decisões e experimentos](docs/DECISOES_E_EXPERIMENTOS.md).

**O que ainda não existe:** sessões e preferências reais, propensão de exposição e correção para viés de posição, avaliações online, otimização listwise, autenticação, banco de dados ou uma implantação de produção. A comparação de seeds e os testes com restaurantes inéditos não resolvem esses problemas.

---

**Sobre o desenvolvimento:** protótipo construído e revisado com assistência de ferramentas de IA. Código, dados gerados e scripts estão disponíveis para auditoria. A capacidade de compreender, reproduzir e evoluir o trabalho importa mais do que tentar esconder as ferramentas utilizadas.

Estudo de sistemas de recomendação · [Lucas Nunes](https://github.com/lucastnunes06) · [Licença MIT](LICENSE).
