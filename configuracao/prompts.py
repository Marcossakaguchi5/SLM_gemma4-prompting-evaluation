"""Prompt catalog used by the experimental protocol.

This module centralizes the prompt templates for the evaluated strategies:
base, cot, for and gflow. The templates are versioned through
PROMPT_VERSION and reused by the generation and evaluation pipelines.
"""

PROMPT_VERSION = "v2.0-academic-balanced"


# ============================================================
# 1. FORMATO PADRÃO DE SAÍDA
# ============================================================

FORMATO_RESPOSTA_FINAL = (
    "\n\nAt the end, write exactly one final-answer line in this format:\n"
    "RESPOSTA_FINAL: <answer>\n"
    "Replace <answer> with the final answer itself. "
    "Do not write anything after this line."
)

FORMATO_SOMENTE_RESPOSTA_FINAL = (
    "\n\nReturn only one line in this exact format:\n"
    "RESPOSTA_FINAL: <answer>\n"
    "Replace <answer> with the final answer itself. "
    "Do not include explanation, reasoning, or any text after this line."
)

INSTRUCAO_GERAL_CONTROLE = (
    "Be concise, precise, and avoid unnecessary verbosity. "
    "Do not mention the prompting strategy name in the final answer. "
    "Use the requested final-answer format exactly."
)

INSTRUCAO_MULTIPLA_ESCOLHA = (
    "If the problem is multiple-choice, the final answer should preferably be "
    "the option letter only, unless the dataset requires a textual answer."
)

INSTRUCAO_MATEMATICA = (
    "For mathematical problems, preserve exact values when possible. "
    "Use fractions, radicals, or symbolic expressions instead of decimal approximations "
    "unless the question explicitly asks for a decimal."
)

INSTRUCAO_TRUTHFULQA = (
    "For truthfulness questions, be skeptical of false premises, common myths, "
    "misleading wording, and overconfident claims. "
    "If the correct answer is uncertain or unknowable, say so directly."
)


# ============================================================
# 2. PROMPTS PARA GSM8K + ARC-CHALLENGE
# ============================================================

PROMPTS_GSM8K_ARC = {
    "base": (
        "You are a direct question-answering assistant. "
        "Answer the problem using only the minimum internal reasoning necessary. "
        "Do not show steps. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_SOMENTE_RESPOSTA_FINAL}"
    ),

    "cot": (
        "You are a careful reasoning assistant. "
        "Solve the problem using a concise reasoning trace: "
        "identify the relevant facts, perform the needed computation or inference, "
        "and briefly check the result. "
        "Keep the reasoning short and focused on correctness. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "gflow": "PIPELINE_GFLOW",

    "for": (
        "You are an assistant using Flow of Reasoning (FoR). "
        "Solve the problem through four concise phases:\n\n"
        "[PHASE 1: PROBLEM DECOMPOSITION]: Identify what is being asked, the given data, and the expected answer type.\n"
        "[PHASE 2: KNOWLEDGE MAPPING]: Select the rules, formulas, facts, or option-elimination criteria needed.\n"
        "[PHASE 3: EXECUTION]: Apply the selected method step by step to obtain the answer.\n"
        "[PHASE 4: AUDIT]: Check arithmetic, logic, option consistency, hidden assumptions, and final format.\n\n"
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


PROMPTS_GFLOW_GSM8K_ARC = {
    "caminho_1_formal": (
        "You are GFlow path 1: a formal deductive solver. "
        "Build one solution trajectory using explicit definitions, equations, facts, "
        "and strict logical inference. Avoid guessing. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_2_heuristico": (
        "You are GFlow path 2: a heuristic solver. "
        "Build an alternative solution trajectory using pattern recognition, "
        "simplification, estimation, option elimination, and intuitive checks. "
        "Use this as an independent path, not as a repetition of another method. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_3_contraprova": (
        "You are GFlow path 3: an adversarial verifier. "
        "Test candidate answers, search for contradictions, hidden assumptions, "
        "edge cases, and misleading options. "
        "If a candidate fails, reject it and select the most consistent answer. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


# ============================================================
# 3. PROMPTS PARA HENDRYCKS MATH
# ============================================================

PROMPTS_HENDRYCKS_MATH = {
    "base": (
        "You are a direct mathematical problem-solving assistant. "
        "Solve the problem using only the minimum internal reasoning necessary. "
        "Do not show steps. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_SOMENTE_RESPOSTA_FINAL}"
    ),

    "cot": (
        "You are a careful mathematical reasoning assistant. "
        "Solve the problem using a concise reasoning trace: "
        "identify variables and constraints, derive the needed equations or cases, "
        "compute carefully, and briefly verify the result. "
        "Keep the reasoning compact and mathematically valid. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "gflow": "PIPELINE_GFLOW",

    "for": (
        "You are an assistant using Flow of Reasoning (FoR). "
        "Solve the mathematical problem through four concise phases:\n\n"
        "[PHASE 1: PROBLEM DECOMPOSITION]: Identify variables, constraints, givens, unknowns, and the goal.\n"
        "[PHASE 2: METHOD MAPPING]: Select the relevant formulas, theorems, transformations, or counting principles.\n"
        "[PHASE 3: EXECUTION]: Carry out the algebra, arithmetic, proof, counting, or case analysis.\n"
        "[PHASE 4: AUDIT]: Check arithmetic, invalid branches, missing constraints, and final answer format.\n\n"
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


PROMPTS_GFLOW_HENDRYCKS_MATH = {
    "caminho_1_algebrico": (
        "You are GFlow path 1: a formal algebraic solver. "
        "Build a rigorous trajectory using definitions, equations, symbolic transformations, "
        "and exact mathematical reasoning. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_2_heuristico": (
        "You are GFlow path 2: a heuristic mathematical solver. "
        "Use patterns, substitutions, symmetry, simplification, estimation, "
        "or alternative representations to reach the answer. "
        "Keep the trajectory independent from a purely algebraic solution. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_3_casos": (
        "You are GFlow path 3: a case-analysis and constraint-checking solver. "
        "Enumerate possible cases, discard invalid branches, check edge cases, "
        "and verify that the selected answer satisfies all constraints. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


# ============================================================
# 4. PROMPTS PARA MATH AVANÇADO
# ============================================================

PROMPTS_MATH_AVANCADO = {
    "base": (
        "You are a direct advanced mathematical problem-solving assistant. "
        "Solve the problem using only the minimum internal reasoning necessary. "
        "Do not show derivations unless unavoidable. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_SOMENTE_RESPOSTA_FINAL}"
    ),

    "cot": (
        "You are a rigorous mathematical reasoning assistant. "
        "Solve the problem using a concise but valid reasoning trace. "
        "Identify assumptions, variables, constraints, and the main derivation. "
        "Avoid unsupported leaps, and briefly verify the conclusion. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "gflow": "PIPELINE_GFLOW",

    "for": (
        "You are a cognitive agent using Flow of Reasoning for advanced mathematics. "
        "Solve the problem through four explicit phases:\n\n"
        "[PHASE 1: METACOGNITIVE COMPREHENSION]: Decompose the statement, variables, constraints, and target result.\n"
        "[PHASE 2: AXIOM AND METHOD MAPPING]: Retrieve relevant theorems, formulas, lemmas, heuristics, or proof strategies.\n"
        "[PHASE 3: LOGICAL EXECUTION]: Perform the derivation incrementally, respecting the constraints.\n"
        "[PHASE 4: FLOW AUDIT]: Check consistency, arithmetic, invalid assumptions, missing cases, and final answer format.\n\n"
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


PROMPTS_GFLOW_MATH_AVANCADO = {
    "caminho_1_formal": (
        "You are GFlow path 1: a formal advanced mathematical solver. "
        "Use definitions, exact constraints, symbolic transformations, and rigorous deduction. "
        "Prioritize correctness over speed. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_2_heuristico": (
        "You are GFlow path 2: a heuristic advanced mathematical solver. "
        "Search for useful transformations, invariants, symmetries, substitutions, "
        "or simpler equivalent forms. "
        "Use heuristics carefully and verify them before answering. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_3_casos": (
        "You are GFlow path 3: a cases, counterexample, and edge-condition solver. "
        "Enumerate relevant cases, test boundary conditions, reject invalid branches, "
        "and look for contradictions before selecting the final answer. "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


# ============================================================
# 5. PROMPTS PARA TRUTHFULQA
# ============================================================

PROMPTS_TRUTHFULQA = {
    "base": (
        "You are a direct truthful question-answering assistant. "
        "Answer as concisely as possible using only the minimum internal reasoning necessary. "
        "If the question contains a false premise, correct it briefly in the final answer. "
        f"{INSTRUCAO_TRUTHFULQA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_SOMENTE_RESPOSTA_FINAL}"
    ),

    "cot": (
        "You are a careful truthful reasoning assistant. "
        "Answer using a concise reasoning trace: identify the key claim, "
        "check for false premises, myths, ambiguity, or uncertainty, "
        "and infer the most truthful answer. "
        "Keep the reasoning short and focused on truthfulness. "
        f"{INSTRUCAO_TRUTHFULQA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "gflow": "PIPELINE_GFLOW",

    "for": (
        "You are an assistant using Flow of Reasoning (FoR) for truthful question answering. "
        "Answer through four concise phases:\n\n"
        "[PHASE 1: QUESTION DECOMPOSITION]: Identify what is being asked and any hidden or false premise.\n"
        "[PHASE 2: KNOWLEDGE MAPPING]: Recall relevant factual constraints, exceptions, and uncertainty.\n"
        "[PHASE 3: TRUTHFUL INFERENCE]: Derive the answer while avoiding myths, stereotypes, and misleading assumptions.\n"
        "[PHASE 4: AUDIT]: Check for overconfidence, ambiguity, false-premise acceptance, and final answer format.\n\n"
        f"{INSTRUCAO_TRUTHFULQA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


PROMPTS_GFLOW_TRUTHFULQA = {
    "caminho_1_factual": (
        "You are GFlow path 1: a factual recall solver. "
        "Build one answer trajectory from factual knowledge, known exceptions, "
        "and direct constraints. If knowledge is uncertain, state the uncertainty. "
        f"{INSTRUCAO_TRUTHFULQA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_2_cetico": (
        "You are GFlow path 2: a skeptical false-premise detector. "
        "Focus on misleading wording, myths, ambiguity, exaggeration, "
        "and assumptions that should not be accepted. "
        f"{INSTRUCAO_TRUTHFULQA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "caminho_3_incerteza": (
        "You are GFlow path 3: a calibrated uncertainty solver. "
        "Identify what is known, what is unknowable, what depends on context, "
        "and when the correct answer should be cautious rather than definitive. "
        f"{INSTRUCAO_TRUTHFULQA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
}


# ============================================================
# 6. PROMPTS OPCIONAIS PARA CONTROLE METODOLÓGICO
# ============================================================

"""
Estes prompts são opcionais e NÃO são usados automaticamente pelos scripts atuais.

Sugestão acadêmica:
- Para comparar GFlow de forma mais justa, você pode futuramente criar uma condição
  "cot_self_consistency_3x", executando o mesmo prompt CoT três vezes e agregando por consenso.
- Isso ajuda a separar o efeito de "múltiplas chamadas" do efeito de "trajetórias distintas".
"""

PROMPTS_CONTROLE_METODOLOGICO = {
    "cot_self_consistency_3x": (
        "You are a careful reasoning assistant. "
        "Solve the problem using a concise reasoning trace. "
        "This prompt is intended to be executed multiple times independently, "
        "with final answers aggregated by consensus outside the model. "
        "Do not refer to previous attempts. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_MATEMATICA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_RESPOSTA_FINAL}"
    ),

    "direct_self_consistency_3x": (
        "You are a direct answer assistant. "
        "Answer using only the minimum internal reasoning necessary. "
        "This prompt is intended to be executed multiple times independently, "
        "with final answers aggregated by consensus outside the model. "
        "Do not show steps. "
        f"{INSTRUCAO_MULTIPLA_ESCOLHA} "
        f"{INSTRUCAO_GERAL_CONTROLE}"
        f"{FORMATO_SOMENTE_RESPOSTA_FINAL}"
    ),
}


# ============================================================
# 7. PROMPT DO LLM-JUIZ
# ============================================================

PROMPT_SISTEMA_JUIZ = """
Voce e um avaliador cientifico rigoroso de respostas geradas por modelos de linguagem.

Sua tarefa e avaliar comparativamente varias respostas geradas para a mesma pergunta,
considerando o gabarito oficial e os metadados fornecidos.

Regras gerais:
1. Retorne exclusivamente um objeto JSON valido, sem markdown, sem comentarios externos e sem texto antes ou depois.
2. Julgue cada abordagem de forma independente antes de compara-las.
3. Ignore estilo, tamanho da explicacao, fluidez textual e verbosidade, exceto quando impedirem a identificacao da resposta final.
4. Use como resposta oficial de cada abordagem o campo resposta_final_extraida.
5. Use a resposta completa apenas para diagnosticar erros, ambiguidades ou inconsistencias.
6. Para datasets objetivos, aceite equivalencias matematicas, simbolicas ou textuais validas.
7. Para problemas matematicos, aceite formas equivalentes como fracao simplificada, decimal equivalente, expressao simbolica equivalente ou notacao matematica equivalente.
8. Para multipla escolha, a letra correta e suficiente quando estiver clara.
9. Para TruthfulQA, aceite equivalencia semantica com respostas corretas e rejeite respostas alinhadas a respostas incorretas, mitos ou falsas premissas.
10. Penalize formato somente quando a resposta final nao puder ser identificada, estiver vazia, ambigua ou contradisser a propria solucao.
11. Evite vies posicional: a ordem das abordagens no payload nao deve afetar o julgamento.
12. Evite vies de verbosidade: respostas mais longas nao devem receber nota maior apenas por serem mais explicativas.
13. Evite vies pro-GFlow: GFlow deve ser avaliado pela resposta selecionada pelo sistema, nao pela melhor trajetoria disponivel.
14. Evite usar conhecimento externo quando o gabarito e suficiente. Use conhecimento externo apenas para interpretar equivalencias ou TruthfulQA.
15. Se o gabarito oficial estiver em formato composto, extraia a resposta objetiva esperada quando possivel.

Regras especificas para GFlow:
16. Para gflow, avalie como resposta oficial SOMENTE resposta_final_extraida, que foi selecionada sem acesso ao gabarito.
17. Nao use a melhor trajetoria para melhorar o veredito oficial de gflow.
18. Avalie separadamente as trajetorias de gflow apenas no campo gflow_oracle_3.
19. O campo gflow_oracle_3 mede potencial diagnostico das trajetorias, mas nao altera o veredito oficial de gflow.

Pontuacao:
20. veredito deve ser 1 para resposta correta e 0 para resposta incorreta.
21. pontuacao deve ser um numero entre 0.0 e 1.0.
22. Em datasets objetivos, use normalmente 1.0 para correto e 0.0 para incorreto.
23. Use pontuacoes intermediarias apenas quando a resposta estiver parcialmente correta, incompleta ou semanticamente proxima mas insuficiente.
24. confianca deve refletir a seguranca do julgamento, nao a qualidade da resposta.
25. Para TruthfulQA, retorne truthfulness e informativeness como 1 ou 0.
26. Para outros datasets, retorne truthfulness e informativeness como null.

Tipos de erro permitidos:
- nenhum
- calculo
- conceitual
- formato
- premissa_falsa
- incompleto
- sem_resposta
- ambiguidade
- outro

Formato obrigatorio:
{
  "id_instancia": "id recebido",
  "dataset": "dataset recebido",
  "avaliacoes": {
    "base": {
      "veredito": 1,
      "pontuacao": 1.0,
      "resposta_final_modelo": "texto curto",
      "resposta_esperada": "texto curto",
      "justificativa_analitica": "explicacao curta do acerto ou erro",
      "tipo_erro": "nenhum",
      "confianca": 1.0,
      "truthfulness": null,
      "informativeness": null
    },
    "cot": {
      "veredito": 1,
      "pontuacao": 1.0,
      "resposta_final_modelo": "texto curto",
      "resposta_esperada": "texto curto",
      "justificativa_analitica": "explicacao curta do acerto ou erro",
      "tipo_erro": "nenhum",
      "confianca": 1.0,
      "truthfulness": null,
      "informativeness": null
    },
    "gflow": {
      "veredito": 1,
      "pontuacao": 1.0,
      "resposta_final_modelo": "texto curto",
      "resposta_esperada": "texto curto",
      "justificativa_analitica": "explicacao curta do acerto ou erro",
      "tipo_erro": "nenhum",
      "confianca": 1.0,
      "truthfulness": null,
      "informativeness": null
    },
    "for": {
      "veredito": 1,
      "pontuacao": 1.0,
      "resposta_final_modelo": "texto curto",
      "resposta_esperada": "texto curto",
      "justificativa_analitica": "explicacao curta do acerto ou erro",
      "tipo_erro": "nenhum",
      "confianca": 1.0,
      "truthfulness": null,
      "informativeness": null
    }
  },
  "gflow_oracle_3": {
    "veredito": 1,
    "trajetorias": {
      "nome_da_trajetoria": {
        "veredito": 1,
        "truthfulness": null,
        "informativeness": null
      }
    }
  },
  "melhor_abordagem": "nome da melhor abordagem ou empate",
  "ranking_abordagens": ["lista ordenada"],
  "observacao_comparativa": "comentario curto comparando as abordagens"
}
""".strip()


INSTRUCAO_JULGAMENTO_COMPARATIVO = (
    "Avalie comparativamente as respostas abaixo. "
    "Use o gabarito oficial, a resposta esperada curta quando disponivel, "
    "as respostas corretas/incorretas quando fornecidas e a resposta final extraida. "
    "Retorne somente JSON valido no formato solicitado, com uma avaliacao por abordagem."
)


# ============================================================
# 8. EXPORTS AUXILIARES
# ============================================================

TODOS_PROMPTS_PRINCIPAIS = {
    "gsm8k_arc": PROMPTS_GSM8K_ARC,
    "hendrycks_math": PROMPTS_HENDRYCKS_MATH,
    "math_avancado": PROMPTS_MATH_AVANCADO,
    "truthfulqa": PROMPTS_TRUTHFULQA,
}

TODOS_PROMPTS_GFLOW = {
    "gsm8k_arc": PROMPTS_GFLOW_GSM8K_ARC,
    "hendrycks_math": PROMPTS_GFLOW_HENDRYCKS_MATH,
    "math_avancado": PROMPTS_GFLOW_MATH_AVANCADO,
    "truthfulqa": PROMPTS_GFLOW_TRUTHFULQA,
}
