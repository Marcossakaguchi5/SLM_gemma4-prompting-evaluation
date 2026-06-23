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

## Juiz externo

Para usar o OpenRouter como juiz, defina no `.env`:

```dotenv
JUDGE_PROVIDER=openrouter
JUDGE_MODEL_NAME=provedor/modelo
OPENROUTER_API_KEY=sua_chave
```

Por exemplo, `JUDGE_MODEL_NAME` deve ser o identificador do modelo escolhido no OpenRouter. A chave fica somente no `.env`, que é ignorado pelo Git; não a adicione ao `.env.example`. Para Gemini, use `JUDGE_PROVIDER=gemini` e `GEMINI_API_KEY` ou `GOOGLE_API_KEY`.
