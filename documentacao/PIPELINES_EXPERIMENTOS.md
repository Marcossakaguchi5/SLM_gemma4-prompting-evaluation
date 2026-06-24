# Pipelines separados e configuração por `.env`

Os pipelines foram separados para que cada etapa tenha uma responsabilidade clara e leia a mesma configuração local de `.env`:

| Arquivo | Responsabilidade |
| --- | --- |
| `pipelines/geracao.py` | Executa os benchmarks e cria uma nova execução isolada. |
| `pipelines/avaliacao.py` | Executa o LLM-as-a-Judge sobre a última execução gerada. |
| `pipelines/graficos.py` | Consolida métricas e produz os gráficos da mesma execução. |

## Configuração

Edite `.env`; há uma cópia segura e versionável em `.env.example`. Os campos mais usados são:

```dotenv
SLM_MODEL_NAME=gemma4:e4b
EXPERIMENT_NUM_SAMPLES=10
EXPERIMENT_SEED=20260612
PIPELINE_EXPERIMENTS=gsm8k_arc,hendrycks_math,truthfulqa
EXPERIMENT_TASK_CONCURRENCY=16
EXPERIMENT_CALL_CONCURRENCY=4
```

Com os três benchmarks principais, esse valor gera 10 questões de Hendrycks MATH, 10 de TruthfulQA e 10 para cada dataset do arquivo GSM8K/ARC: 40 instâncias no total. Como cada instância é avaliada por `base`, `cot`, `for` e `gflow`, a rodada armazena 160 respostas oficiais, além dos rastros das três trajetórias do GFlow.

`JUDGE_SEED` e `ANALYSIS_SEED` ficam vazias por padrão para reutilizar automaticamente `EXPERIMENT_SEED`. Preencha-as somente se quiser uma seed específica para essas etapas.

Para incluir a arena opcional, adicione `math_avancado` em `PIPELINE_EXPERIMENTS`.

## Nemotron Diffusion como baseline externo

O Nemotron participa como **outro modelo**, e nao como uma quinta abordagem de prompting do Gemma. Isso evita atribuir a uma estrategia de prompt uma diferenca que vem da arquitetura do modelo. Ele usa o prompt `base` adequado a cada dataset e fica identificado pelo modelo `nvidia/Nemotron-Labs-Diffusion-3B` nas tabelas e graficos.

Para a rodada completa com 100 perguntas por dataset, deixe no `.env`:

```dotenv
EXPERIMENT_NUM_SAMPLES=100
```

O pipeline de geracao sempre salva duas amostras separadas na rodada:

```text
amostras/perguntas_amostradas.json
amostras/referencias_amostradas.json
```

O primeiro arquivo contem somente as perguntas e metadados de amostragem; o segundo contem os gabaritos e referencias para avaliacao. Isso impede que a rodada do Nemotron tenha de carregar os resultados do Gemma ou reamostrar os datasets.

Depois da geracao, execute somente o Nemotron com:

```powershell
python -m pipelines.nemotron
```

Ele le `perguntas_amostradas.json`, carrega o modelo uma unica vez e responde sequencialmente aos 400 itens. Suas saidas ficam em `geracao/nemotron_diffusion/`. Em seguida, os pipelines de avaliacao e graficos incluem ambos os modelos automaticamente. A analise separa `modelo / condicao` nas tabelas e gera `comparacao_pareada_modelos_base.csv`, que compara os baselines dos modelos nas mesmas perguntas.

O Nemotron usa PyTorch/Transformers, um caminho independente do Ollama; nao use o paralelismo de requisicoes do Ollama para ele. Neste ambiente, a verificacao encontrou `torch 2.12.1+cpu` e `torch.cuda.is_available() == False`; logo ele usaria CPU, nao a RX 9070 XT. Instale uma pilha PyTorch compativel com a aceleracao desejada antes da rodada completa. O primeiro uso tambem baixa o modelo do Hugging Face.

## Perfil de paralelismo

O `.env` já contém um perfil de throughput para Ryzen 7 9800X3D, RX 9070 XT de 16 GB e 32 GB de RAM:

- 16 tarefas assíncronas para aproveitar os 16 threads lógicos do processador;
- 4 inferências simultâneas, um valor agressivo e mais estável para uma única GPU de 16 GB;
- contexto de 8192 tokens, um modelo carregado por vez e modelo mantido em memória por 30 minutos.

`OLLAMA_*` configura o **servidor** Ollama, não apenas os scripts Python. Após alterar esses campos, reinicie o processo/serviço que executa `ollama serve` com as variáveis do `.env` aplicadas. Para iniciar o servidor a partir desse arquivo, após encerrar uma instância já ativa, execute:

```powershell
.\scripts\iniciar_ollama_com_env.ps1
```

Não force `OLLAMA_LLM_LIBRARY`: deixe o Ollama detectar automaticamente a melhor biblioteca para a RX 9070 XT.

Se houver VRAM livre durante a execução, aumente juntos `EXPERIMENT_CALL_CONCURRENCY` e `OLLAMA_NUM_PARALLEL` de `4` para `5`. Se a taxa de tokens cair, houver troca para RAM ou erros de memória, retorne a `4` ou reduza para `3`.

## Execução

```powershell
python -m pipelines.geracao
# Opcional: executa apenas o Nemotron na amostra ja congelada.
python -m pipelines.nemotron
python -m pipelines.avaliacao
python -m pipelines.graficos
```

O primeiro pipeline grava `ultima_execucao.txt` no diretório definido por `PIPELINE_OUTPUT_ROOT`; os dois seguintes usam esse apontador. Para processar uma rodada específica, passe `--execucao-dir` aos pipelines de avaliação ou gráficos.

Com a configuração padrão, todas as saídas ficam centralizadas em uma nova pasta `resultados/rodada_YYYYMMDD_HHMMSS/`:

```text
resultados/
├── ultima_execucao.txt
└── rodada_YYYYMMDD_HHMMSS/
    ├── manifesto_pipeline.json
    ├── geracao/
    ├── avaliacao_juiz/
    ├── analise/
    └── graficos/
```

Use `--dry-run` em qualquer um dos três módulos para conferir os comandos antes de executar chamadas reais.

Se uma rodada já foi julgada e você quiser repetir somente a avaliação com outro modelo juiz ou configuração, sem gerar novamente as respostas, use:

```powershell
python -m pipelines.avaliacao --refazer
```

Esse comando preserva o julgamento anterior no manifesto e grava a nova avaliação em uma pasta `avaliacao_juiz_reavaliacao_<timestamp>/`.

## Juiz externo

Para usar o OpenRouter como juiz, defina no `.env`:

```dotenv
JUDGE_PROVIDER=openrouter
JUDGE_MODEL_NAME=provedor/modelo
OPENROUTER_API_KEY=sua_chave
```

Por exemplo, `JUDGE_MODEL_NAME` deve ser o identificador do modelo escolhido no OpenRouter. A chave fica somente no `.env`, que é ignorado pelo Git; não a adicione ao `.env.example`. Para Gemini, use `JUDGE_PROVIDER=gemini` e `GEMINI_API_KEY` ou `GOOGLE_API_KEY`.

## Avaliação por tipo de dataset

O pipeline de avaliação executa primeiro `avaliar_deterministico.py` sobre as respostas já geradas. Ele produz um artefato auditável e aplica:

| Dataset | Avaliação principal | Uso do LLM-juiz |
| --- | --- | --- |
| GSM8K | Normalização e comparação numérica. | Somente se o parse falhar. |
| ARC-Challenge | Extração e comparação da alternativa. | Somente se a letra não puder ser extraída. |
| Hendrycks MATH | Exact match, comparação numérica e equivalência simbólica. | Somente se não houver equivalência verificável por código. |
| TruthfulQA | Avaliação semântica. | Principal. |

Com `JUDGE_ONLY_DETERMINISTIC_FALLBACK=true`, o LLM-juiz recebe apenas esses casos pendentes e TruthfulQA. O processamento usa o veredito determinístico como métrica principal dos datasets objetivos e registra a fonte de cada avaliação em `respostas_normalizadas.csv`.

Para executar somente essa auditoria, sem chamar o LLM-juiz, use os diretórios brutos de uma rodada:

```powershell
python avaliar_deterministico.py `
  resultados/rodada_YYYYMMDD_HHMMSS/geracao/gsm8k_arc `
  resultados/rodada_YYYYMMDD_HHMMSS/geracao/hendrycks_math `
  resultados/rodada_YYYYMMDD_HHMMSS/geracao/truthfulqa
```

Sem `--output-root`, a saída é salva em `avaliacoes_deterministicas/rodada_YYYYMMDD_HHMMSS/`, como nas versões anteriores. Quando chamado por `python -m pipelines.avaliacao`, o artefato é salvo dentro da própria rodada em `resultados/rodada_.../avaliacao_deterministica/`.
