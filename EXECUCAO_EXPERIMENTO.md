# Execução do experimento

O pipeline não carrega arquivos `.env`. Informe configurações diretamente no ambiente do processo.

## 1. Dependências

```bash
python3 -m pip install -r requirements.txt
```

O Ollama deve estar ativo e o modelo alvo deve estar instalado.

## 2. Geração

Os scripts usam concorrência 8 e seed padrão `20260612`.

```bash
EXPERIMENT_SEED=20260612 SLM_MODEL_NAME=gemma4:e4b python3 experimento_gsm8k_arc.py
EXPERIMENT_SEED=20260612 SLM_MODEL_NAME=gemma4:e4b python3 experimento_hendrycks_math.py
EXPERIMENT_SEED=20260612 SLM_MODEL_NAME=gemma4:e4b python3 experimento_truthfulqa.py
```

`experimento_math_avancado.py` é uma variante opcional.

## 3. Juiz externo

O juiz primário deve ser diferente do modelo avaliado. A auditoria posicional, habilitada por padrão, julga cada instância na ordem inicial e na ordem inversa.

Exemplo com Gemini e segundo juiz em 10% da amostra:

```bash
GEMINI_API_KEY=... python3 avaliar_llm_judge.py \
  resultados_gsm8k_arc resultados_hendrycks_math resultados_truthfulqa \
  --provider gemini \
  --judge-model MODELO_JUIZ_PRIMARIO \
  --secondary-provider gemini \
  --secondary-judge-model MODELO_JUIZ_SECUNDARIO \
  --secondary-sample-rate 0.10
```

O veredito oficial de `gflow` considera somente a resposta selecionada por consenso sem gabarito. As três trajetórias são avaliadas separadamente apenas para `Oracle@3`.

## 4. Avaliação humana

```bash
python3 gerar_amostra_avaliacao_humana.py \
  resultados_gsm8k_arc resultados_hendrycks_math resultados_truthfulqa \
  --instancias 40
```

Preencha `avaliacao_humana/amostra_avaliacao_humana.csv` sem consultar a chave JSON.

## 5. Consolidação

```bash
python3 processar_resultados.py \
  resultados_gsm8k_arc resultados_hendrycks_math resultados_truthfulqa \
  --avaliacoes avaliacoes_llm_judge \
  --avaliacao-humana avaliacao_humana/amostra_avaliacao_humana.csv \
  --chave-humana avaliacao_humana/chave_avaliacao_humana.json \
  --bootstrap-iterations 5000
```

Principais saídas:

- `Exact Match` e `Answer Match`;
- equivalência simbólica em matemática;
- Truthfulness e Informativeness;
- Oracle@3;
- Win/Tie/Loss e McNemar exato com correção de Holm;
- intervalo de confiança por bootstrap;
- Agreement e Cohen's Kappa;
- Position Bias Rate;
- Accuracy per Second e Gain per Extra Call.

## 6. Gráficos

```bash
python3 gerar_graficos_resultados.py analises_resultados/rodada_YYYYMMDD_HHMMSS
```
