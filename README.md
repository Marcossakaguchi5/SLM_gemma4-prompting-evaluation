# SLM Gemma 4 Prompting Evaluation

Este repositório contém o protocolo experimental usado para avaliar estratégias de prompting em um Small Language Model local executado via Ollama. O objetivo é comparar respostas diretas, Chain-of-Thought, Flow of Reasoning e uma estratégia multi-trajetória inspirada em GFlowNets, mantendo fixa a amostra de perguntas, a seed e o modelo-alvo.

## Objetivo

O estudo mede se estratégias estruturadas de raciocínio melhoram a correção final em relação a uma resposta direta no Gemma 4. A avaliação combina matching determinístico, equivalência simbólica e LLM-as-a-Judge quando a resposta exige julgamento semântico ou quando o avaliador por código não consegue decidir.

## Modelo e estratégias

- Modelo principal: `gemma4:e4b`, via Ollama.
- Baseline externo: `nvidia/Nemotron-Labs-Diffusion-3B`, executado separadamente via PyTorch/Transformers.
- Estratégias avaliadas no Gemma:
  - `base`: resposta direta.
  - `cot`: Chain-of-Thought compacto.
  - `for`: Flow of Reasoning em fases.
  - `gflow`: três trajetórias independentes com seleção/agregação.

O Nemotron não é tratado como uma quinta estratégia de prompting; ele é reportado como comparação externa de modelo usando a mesma amostra congelada.

## Datasets

A rodada principal usa quatro datasets:

- GSM8K
- ARC-Challenge
- Hendrycks MATH/AIME
- TruthfulQA

Por padrão, `.env.example` usa 10 instâncias por dataset para teste rápido. A configuração usada no artigo usa 100 instâncias por dataset; veja `.env.article.example`.

## Como reproduzir

1. Crie e ative um ambiente virtual.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Copie uma configuração de exemplo.

```powershell
Copy-Item .env.example .env
```

Para reproduzir a escala do artigo:

```powershell
Copy-Item .env.article.example .env
```

3. Instale o modelo no Ollama e inicie o servidor.

```powershell
ollama pull gemma4:e4b
.\scripts\iniciar_ollama_com_env.ps1
```

4. Execute o fluxo principal.

```powershell
python -m pipelines.geracao
python -m pipelines.nemotron
python -m pipelines.avaliacao
python -m pipelines.graficos
```

Se uma etapa falhar, use `--retomar --execucao-dir resultados/rodada_YYYYMMDD_HHMMSS` nos pipelines de geração, Nemotron ou avaliação.

## Resultados do artigo

Os resultados brutos são gerados em `resultados/`, pasta ignorada pelo Git por conter artefatos grandes e reprodutíveis. A rodada final descrita no artigo é identificada como `rodada_20260624_040601`.

As figuras usadas no artigo foram versionadas em `figuras/`:

- `figuras/fluxo_experimental_ilustrado.png`
- `figuras/delta_vs_base_gemma_limpo.png`
- `figuras/acuracia_final_por_dataset_abordagem.png`
- `figuras/oracle_gap_gflow.png`

O texto principal está em `artigo_sbc_reformatado_v3_expandido_avaliacoes_academico.tex`.

## Como citar

Se este repositório for usado como base, cite o artigo associado e informe a versão do commit, a seed experimental e a configuração do arquivo `.env` usada na reprodução.
