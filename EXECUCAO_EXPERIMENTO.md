# Execução do experimento

## 1. Dependências

```powershell
python -m pip install -r requirements.txt
```

O Ollama deve estar ativo e o modelo definido em `SLM_MODEL_NAME` deve estar instalado localmente.

## 2. Configuração

Toda a configuração local fica em `.env`; use `.env.example` como referência. Os pontos principais são modelo, tamanho da amostra, seed, seleção de benchmarks e paralelismo. A configuração atual usa 10 perguntas por dataset e salva cada nova rodada em `resultados/rodada_<timestamp>/`.

O perfil incluído foi ajustado para Ryzen 7 9800X3D, RX 9070 XT de 16 GB e 32 GB de RAM:

```dotenv
EXPERIMENT_TASK_CONCURRENCY=16
EXPERIMENT_CALL_CONCURRENCY=4
OLLAMA_NUM_PARALLEL=4
```

Após alterar qualquer variável `OLLAMA_*`, reinicie o servidor Ollama para que ela tenha efeito. Após encerrar uma instância já ativa, use `.\scripts\iniciar_ollama_com_env.ps1` para iniciá-lo com as variáveis do `.env`. Consulte `documentacao/PIPELINES_EXPERIMENTOS.md` para a explicação do perfil e das variáveis.

## 3. Pipelines

```powershell
python -m pipelines.geracao
python -m pipelines.nemotron  # opcional: executa somente o Nemotron
python -m pipelines.avaliacao
python -m pipelines.graficos
```

Os scripts usam a mesma execução: a geração cria uma pasta em `PIPELINE_OUTPUT_ROOT`, e os dois pipelines seguintes consomem automaticamente a última execução criada. A geração, a avaliação e os gráficos permanecem separados para que seja possível inspecionar cada etapa.

Para validar a configuração sem chamar modelos ou baixar datasets:

```powershell
python -m pipelines.geracao --dry-run
```

## 4. Avaliação humana opcional

```powershell
python gerar_amostra_avaliacao_humana.py <DIRETORIO_DE_RESULTADOS> --instancias 40
```

Depois de preencher o CSV, informe `HUMAN_EVALUATION_CSV` e `HUMAN_EVALUATION_KEY` no `.env` antes de executar `python -m pipelines.graficos`.
