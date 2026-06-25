# Conclusões da rodada experimental

## Escopo da evidência

Esta rodada avaliou o modelo local `gemma4:e4b` com prompts `v2.0-academic-balanced`. Foram usadas 10 instâncias de cada domínio: GSM8K, ARC-Challenge, Hendrycks MATH e TruthfulQA. Isso resultou em 40 perguntas e 160 respostas oficiais (quatro abordagens por pergunta). O julgamento foi realizado por `z-ai/glm-5.2` via OpenRouter, com auditoria de ordem normal e invertida.

As conclusões abaixo são **preliminares**: a amostra por dataset é pequena (`n=10`) e não houve segundo juiz nem avaliação humana nesta rodada.

## Conclusão principal

O **Chain-of-Thought (`cot`)** foi a melhor estratégia desta amostra: obteve 87,5% de acurácia global, contra 82,5% para `base`, `for` e `gflow`. O ganho foi de 5 pontos percentuais sobre a linha de base, com custo de aproximadamente 15% a mais de tempo médio por resposta.

Entretanto, a evidência ainda não permite afirmar superioridade estatística. Os testes pareados de McNemar apresentaram `p=1,0` para todas as comparações e os intervalos de confiança são largos, consequência direta do tamanho amostral reduzido.

## Resultado por abordagem

| Abordagem | Acurácia | IC 95% | Tempo médio | Chamadas SLM por pergunta | Leitura |
| --- | ---: | --- | ---: | ---: | --- |
| `base` | 82,5% | 70,0%–92,5% | 182,5 s | 1 | Melhor referência de eficiência. |
| `cot` | **87,5%** | 77,5%–97,5% | 209,6 s | 1 | Melhor qualidade observada, com sobrecusto moderado. |
| `for` | 82,5% | 70,0%–92,5% | 204,3 s | 1 | Não trouxe ganho sobre a base e foi mais lento. |
| `gflow` | 82,5% | 70,0%–92,5% | 268,2 s | 3 | Não superou a base após a seleção por consenso. |

## Resultado por dataset

- **GSM8K:** todas as abordagens acertaram 10 de 10. Esta faixa não diferencia as estratégias nesta amostra.
- **ARC-Challenge:** `cot` acertou 10 de 10; as demais acertaram 9 de 10. Há um sinal favorável ao CoT, mas apenas uma instância separa os métodos.
- **Hendrycks MATH:** todas as abordagens acertaram 7 de 10. O domínio foi o mais difícil e nenhum método mostrou vantagem nesta amostra.
- **TruthfulQA:** `cot` atingiu 8 de 10, contra 7 de 10 para as demais. Também teve a maior informativeness (70%, contra 60% das outras abordagens); a truthfulness foi 80% para `base`, `cot` e `gflow`, e 70% para `for`.

## Multi-trajetória GFlow

O `gflow` utilizou três chamadas por pergunta, produziu três vezes mais chamadas ao SLM e aproximadamente 4,6 vezes mais tokens de saída do que a base. Mesmo assim, sua acurácia oficial permaneceu em 82,5%, igual à base.

O diagnóstico `Oracle@3` foi 85,0%: em pelo menos uma pergunta adicional, uma trajetória individual acertou, mas o consenso não selecionou essa resposta. Portanto, o limite potencial das trajetórias é ligeiramente maior que a seleção atual, mas o ganho potencial observado é de apenas uma resposta em 40. Nesta rodada, o custo adicional do GFlow não se justifica pela acurácia oficial.

## Eficiência

`base` manteve a maior acurácia por segundo (0,004521). O CoT ficou próximo (0,004175), enquanto FoR (0,004039) e GFlow (0,003075) perderam eficiência. Assim, a escolha prática atual é:

- usar **CoT** quando a prioridade for a qualidade observada;
- usar **base** quando latência e custo forem prioritários;
- não adotar **FoR** ou **GFlow** como padrão com base apenas nesta rodada.

## Confiabilidade do julgamento

Não houve falhas operacionais nas 160 respostas nem nos 160 julgamentos. A auditoria posicional apresentou 97,5% de concordância entre ordem normal e invertida (`Cohen's kappa = 0,9135`), equivalente a 2,5% de divergência. O viés posicional foi nulo nos três datasets objetivos e de 10% em TruthfulQA.

Isso sugere julgamento estável para esta rodada, mas não elimina a necessidade de validar uma amostra com segundo juiz e avaliação humana, especialmente para TruthfulQA.

## Próximos passos recomendados

1. Repetir o protocolo principal com 100 instâncias por dataset, mantendo modelo, prompts, seed e configuração de inferência registrados.
2. Tratar CoT como candidato principal e comparar novamente CoT versus base com McNemar e intervalos de confiança mais estreitos.
3. Incluir um segundo juiz e uma amostra humana para verificar a avaliação semântica, sobretudo em TruthfulQA.
4. Investigar a seleção do GFlow antes de aumentar seu orçamento: o `Oracle@3` mostra que existe uma trajetória correta que o consenso não escolheu em pelo menos um caso.
