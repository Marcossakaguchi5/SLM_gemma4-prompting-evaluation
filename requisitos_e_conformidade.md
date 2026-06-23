# Requisitos e conformidade

Levantamento baseado em `artigo_sbc_reformatado_v3_expandido.tex` e verificação dos scripts do projeto.

## Requisitos funcionais

| ID | Requisito | Situação | Evidência |
|---|---|---|---|
| RF01 | Carregar 100 instâncias de GSM8K, ARC-Challenge, Hendrycks MATH/AIME e TruthfulQA por amostragem reproduzível. | Atendido | Seed persistida; amostragem estratificada por tipo/categoria quando disponível. |
| RF02 | Executar o SLM local via Ollama e permitir troca por `SLM_MODEL_NAME`. | Atendido | Todos os scripts de experimento usam `ChatOllama` e a variável de ambiente. |
| RF03 | Comparar `base`, `cot`, `for` e `gflow` sobre as mesmas perguntas. | Atendido | Dicionário `PROMPTS` dos scripts de experimento. |
| RF04 | Gerar três trajetórias independentes no `gflow` e selecionar sem gabarito. | Atendido | Voto majoritário normalizado e desempate por prioridade determinística. |
| RF05 | Adaptar os papéis do `gflow` ao domínio avaliado. | Atendido | Formal/heurístico/casos em matemática e factual/cético/incerteza em TruthfulQA. |
| RF06 | Exigir o marcador `RESPOSTA_FINAL` nas respostas. | Atendido | Constante `FORMATO_RESPOSTA_FINAL` e função `extrair_resposta_final`. |
| RF07 | Persistir cada rodada em JSON consolidado, JSONL parcial e log. | Atendido | Funções de preparação e persistência dos quatro scripts de geração. |
| RF08 | Registrar pergunta, gabarito, modelo, dataset, abordagem, resposta, rastros, status e duração. | Atendido | Objetos retornados por `processar_instancia`. |
| RF09 | Agrupar no juiz todas as abordagens da mesma instância, dataset e modelo. | Atendido | `agrupar_por_instancia` em `avaliar_llm_judge.py`. |
| RF10 | Julgar comparativamente com ordem embaralhada e repetir em ordem reversa. | Atendido | Auditoria posicional habilitada por padrão em `avaliar_llm_judge.py`. |
| RF11 | Produzir veredito, pontuação, Truthfulness, Informativeness, justificativa, confiança, ranking e Oracle@3. | Atendido | Esquema estruturado do juiz externo. |
| RF12 | Não transformar falha do juiz em resposta errada do SLM. | Atendido após correção | Falhas usam `veredito: null`, `status_avaliacao` e contagem separada. |
| RF13 | Consolidar métricas objetivas, semânticas, estatísticas e de eficiência. | Atendido | Exact/Answer Match, equivalência simbólica, bootstrap, Win/Tie/Loss, McNemar exato, Holm e custo. |
| RF14 | Preservar o modelo nas avaliações e comparações de execuções com vários SLMs. | Atendido após correção | Chaves, deltas e comparação pareada incluem `modelo`. |
| RF15 | Gerar gráficos de acurácia, duração, delta e custo versus acurácia. | Atendido | `gerar_graficos_resultados.py`, com PNG ou fallback SVG. |
| RF16 | Oferecer uma arena matemática avançada opcional. | Atendido | `experimento_math_avancado.py`; não entra nas 400 instâncias principais. |
| RF17 | Medir concordância com segundo juiz e avaliação humana. | Atendido | Segundo juiz em amostra, gerador cego de avaliação humana, Agreement e Cohen's Kappa. |
| RF18 | Medir viés posicional do juiz. | Atendido | Position Bias Rate calculado pela divergência entre ordem original e inversa. |
| RF19 | Medir o teto da estratégia multi-trajetória. | Atendido | Oracle@3 objetivo ou semântico, separado da resposta oficial de `gflow`. |
| RF20 | Medir eficiência. | Atendido | Accuracy per Second e Gain per Extra Call. |

## Requisitos não funcionais

| ID | Requisito | Situação | Evidência |
|---|---|---|---|
| RNF01 | Limitar a concorrência a oito tarefas ativas e oito chamadas simultâneas ao SLM. | Atendido | Há semáforos separados para tarefas e para cada `ainvoke`, inclusive nas trajetórias de `gflow`. |
| RNF02 | Usar temperatura 0.0 e `top_p` 0.9 para reduzir variação. | Atendido | Configuração de `ChatOllama` nos scripts. |
| RNF03 | Manter auditabilidade e recuperação parcial em execuções longas. | Atendido | JSONL gravado item a item, JSON consolidado e logs por rodada. |
| RNF04 | Separar geração, julgamento, análise e gráficos em módulos e pastas. | Atendido | Organização dos scripts e diretórios de saída. |
| RNF05 | Usar um juiz separado e preferencialmente mais forte que o SLM alvo. | Atendido com configuração externa | `--judge-model`/`JUDGE_MODEL_NAME` é obrigatório; o mesmo modelo Ollama é bloqueado por padrão. |
| RNF06 | Não acessar arquivo `.env`. | Atendido | Não há carregamento de dotenv; as configurações são lidas apenas do ambiente do processo. |
| RNF07 | Registrar reprodutibilidade e recursos de hardware. | Atendido | Manifesto por rodada, hashes de prompts, versões, seed, IDs e monitor de RAM/VRAM. |
| RNF08 | Padronizar o idioma dos prompts. | Atendido | Prompts executados pelo SLM estão em inglês. |

## Observações

- A seleção usa seed padrão `20260612`, substituível por `EXPERIMENT_SEED`.
- `experimento_math_avancado.py` repete o domínio Hendrycks MATH com prompts mais fortes e deve ser tratado como experimento adicional.
- A execução integrada depende dos pacotes de `requirements.txt`, dos datasets externos e de um servidor Ollama/modelo disponível.
- A extração de `\boxed{...}` foi ajustada para aceitar expressões com chaves aninhadas, como `\boxed{\frac{1}{2}}`.
- A auditoria por segundo juiz depende de `--secondary-judge-model`; a avaliação humana depende do preenchimento do CSV cego.
