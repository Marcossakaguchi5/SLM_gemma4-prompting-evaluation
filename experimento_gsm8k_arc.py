import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from configuracao.ambiente import inteiro, texto
from configuracao.prompts import (
    PROMPT_VERSION,
    PROMPTS_GFLOW_GSM8K_ARC as PROMPTS_GFLOW_AGENTES,
    PROMPTS_GSM8K_ARC as PROMPTS,
)
from util_experimento import (
    MonitorRecursos,
    adicionar_jsonl_duravel,
    amostrar_reprodutivel,
    carregar_checkpoint_jsonl,
    chave_checkpoint_resultado,
    extrair_telemetria_resposta,
    preparar_diretorio_checkpoint,
    salvar_json_atomico,
    salvar_manifesto,
    seed_experimento,
    selecionar_trajetoria_consenso,
    somar_telemetrias,
)

# ==========================================
# 1. CONFIGURACOES E INICIALIZACAO
# ==========================================
MODEL_NAME = texto("SLM_MODEL_NAME", "gemma4:e4b")
TASK_CONCURRENCY_LIMIT = inteiro("EXPERIMENT_TASK_CONCURRENCY", 16, minimo=1)
CALL_CONCURRENCY_LIMIT = inteiro("EXPERIMENT_CALL_CONCURRENCY", 4, minimo=1)
NUM_AMOSTRAS_POR_DATASET = inteiro("EXPERIMENT_NUM_SAMPLES", 100, minimo=1)
EXPERIMENT_SEED = seed_experimento()

OUTPUT_ROOT = Path(texto("EXPERIMENT_OUTPUT_ROOT", "resultados_gsm8k_arc"))
OUTPUT_FILE_NAME = "resultados_gsm8k_arc.json"
PARTIAL_OUTPUT_FILE_NAME = "resultados_gsm8k_arc.parcial.jsonl"
LOG_FILE_NAME = "experimento_gsm8k_arc.log"

RUN_DIR = None
OUTPUT_FILE = None
PARTIAL_OUTPUT_FILE = None
LOG_FILE = None
RETOMANDO = False

logger = logging.getLogger("experimento_gsm8k_arc")
llm = ChatOllama(model=MODEL_NAME, temperature=0.0, top_p=0.9)

ABORDAGEM_ORDEM = {
    "base": 0,
    "cot": 1,
    "gflow": 2,
    "for": 3,
}


# ==========================================
# 2. LOGS E PERSISTENCIA PARCIAL
# ==========================================
def preparar_diretorio_rodada():
    return preparar_diretorio_checkpoint(OUTPUT_ROOT)


def configurar_arquivos_saida():
    global RUN_DIR, OUTPUT_FILE, PARTIAL_OUTPUT_FILE, LOG_FILE, RETOMANDO

    RUN_DIR, RETOMANDO = preparar_diretorio_rodada()
    OUTPUT_FILE = RUN_DIR / OUTPUT_FILE_NAME
    PARTIAL_OUTPUT_FILE = RUN_DIR / PARTIAL_OUTPUT_FILE_NAME
    LOG_FILE = RUN_DIR / LOG_FILE_NAME


def configurar_logging():
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def texto_para_log(valor):
    if isinstance(valor, str):
        return valor
    return json.dumps(valor, ensure_ascii=False)


def salvar_resultado_parcial(resultado):
    adicionar_jsonl_duravel(PARTIAL_OUTPUT_FILE, resultado)


def ordenar_resultados(resultados):
    return sorted(
        resultados,
        key=lambda item: (
            item["id_instancia"],
            ABORDAGEM_ORDEM.get(item["abordagem"], 999),
        ),
    )


def salvar_resultados_consolidados(resultados):
    salvar_json_atomico(OUTPUT_FILE, ordenar_resultados(resultados), indent=4)


def carregar_checkpoint():
    return carregar_checkpoint_jsonl(PARTIAL_OUTPUT_FILE, chave_checkpoint_resultado)


GFLOW_TRAJETORIAS = [
    ("caminho_1_formal", "CAMINHO 1 - FORMAL"),
    ("caminho_2_heuristico", "CAMINHO 2 - HEURISTICO"),
    ("caminho_3_contraprova", "CAMINHO 3 - CONTRAPROVA"),
]


# ==========================================
# 4. CARREGAMENTO DOS DATASETS
# ==========================================
def carregar_dados_amostra():
    logger.info("Carregando subconjuntos dos datasets do HuggingFace...")

    dataset_gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    selecionados_gsm8k = amostrar_reprodutivel(
        dataset_gsm8k,
        NUM_AMOSTRAS_POR_DATASET,
        EXPERIMENT_SEED,
    )
    amostra_gsm8k = [
        {
            "id": f"gsm8k_{indice_original:04d}",
            "indice_original": indice_original,
            "dataset": "GSM8K",
            "pergunta": item["question"],
            "gabarito": item["answer"],
        }
        for indice_original, item in selecionados_gsm8k
    ]
    logger.info("GSM8K carregado: %s instancias.", len(amostra_gsm8k))

    dataset_arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    amostra_arc = []
    selecionados_arc = amostrar_reprodutivel(
        dataset_arc,
        NUM_AMOSTRAS_POR_DATASET,
        EXPERIMENT_SEED + 1,
    )
    for indice_original, item in selecionados_arc:
        opcoes = "\n".join(
            [
                f"{label}) {texto}"
                for label, texto in zip(item["choices"]["label"], item["choices"]["text"])
            ]
        )
        pergunta_formatada = f"{item['question']}\nOptions:\n{opcoes}"
        amostra_arc.append(
            {
                "id": f"arc_{indice_original:04d}",
                "indice_original": indice_original,
                "dataset": "ARC-Challenge",
                "pergunta": pergunta_formatada,
                "gabarito": item["answerKey"],
            }
        )
    logger.info("ARC-Challenge carregado: %s instancias.", len(amostra_arc))

    return amostra_gsm8k + amostra_arc


# ==========================================
# 5. EXECUCAO DA INFERENCIA
# ==========================================
async def executar_trajetoria_gflow(sem_chamadas, chave_prompt, pergunta):
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", PROMPTS_GFLOW_AGENTES[chave_prompt]),
                ("human", "{input}"),
            ]
        )
        | llm
    )
    async with sem_chamadas:
        resposta = await chain.ainvoke({"input": pergunta})
    return chave_prompt, resposta.content, extrair_telemetria_resposta(resposta)

async def executar_gflow(sem_chamadas, pergunta):
    respostas = await asyncio.gather(
        *[
            executar_trajetoria_gflow(sem_chamadas, chave_prompt, pergunta)
            for chave_prompt, _ in GFLOW_TRAJETORIAS
        ]
    )
    respostas_por_chave = {
        chave: conteudo for chave, conteudo, _ in respostas
    }
    telemetrias_por_chave = {
        chave: telemetria for chave, _, telemetria in respostas
    }
    rastros = {}

    for chave_prompt, _ in GFLOW_TRAJETORIAS:
        conteudo = respostas_por_chave[chave_prompt]
        rastros[f"gflow_{chave_prompt}"] = conteudo

    resposta_agregada, selecao = selecionar_trajetoria_consenso(
        respostas_por_chave,
        [chave for chave, _ in GFLOW_TRAJETORIAS],
    )
    return (
        resposta_agregada,
        rastros,
        selecao,
        somar_telemetrias(list(telemetrias_por_chave.values())),
    )


async def processar_instancia(
    sem_tarefas,
    sem_chamadas,
    item,
    abordagem,
    prompt_sistema,
):
    async with sem_tarefas:
        return await executar_instancia(
            sem_chamadas,
            item,
            abordagem,
            prompt_sistema,
        )


async def executar_instancia(sem_chamadas, item, abordagem, prompt_sistema):
    inicio = time.perf_counter()
    logger.info(
        "Iniciando inferencia | id=%s | dataset=%s | abordagem=%s",
        item["id"],
        item["dataset"],
        abordagem,
    )

    try:
        rastros_execucao = {}
        selecao_gflow = {}
        if abordagem == "gflow":
            output_text, rastros_execucao, selecao_gflow, telemetria = await executar_gflow(
                sem_chamadas,
                item["pergunta"],
            )
            numero_chamadas = 3
        else:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt_sistema),
                    ("human", "{input}"),
                ]
            )
            chain = prompt_template | llm
            async with sem_chamadas:
                response = await chain.ainvoke({"input": item["pergunta"]})
            output_text = response.content
            telemetria = extrair_telemetria_resposta(response)
            numero_chamadas = 1

        duracao = round(time.perf_counter() - inicio, 2)
        return {
            "id_instancia": item["id"],
            "modelo": MODEL_NAME,
            "dataset": item["dataset"],
            "indice_original": item["indice_original"],
            "seed_amostragem": EXPERIMENT_SEED,
            "abordagem": abordagem,
            "ordem_abordagem": ABORDAGEM_ORDEM.get(abordagem),
            "pergunta": item["pergunta"],
            "gabarito_oficial": item["gabarito"],
            "resposta_gerada": output_text,
            "rastros_execucao": rastros_execucao,
            "selecao_gflow": selecao_gflow,
            "numero_chamadas_slm": numero_chamadas,
            "telemetria": telemetria,
            "status": "ok",
            "duracao_segundos": duracao,
        }
    except Exception as exc:
        duracao = round(time.perf_counter() - inicio, 2)
        logger.exception(
            "Erro ao processar id=%s | dataset=%s | abordagem=%s",
            item["id"],
            item["dataset"],
            abordagem,
        )
        return {
            "id_instancia": item["id"],
            "modelo": MODEL_NAME,
            "dataset": item["dataset"],
            "indice_original": item["indice_original"],
            "seed_amostragem": EXPERIMENT_SEED,
            "abordagem": abordagem,
            "ordem_abordagem": ABORDAGEM_ORDEM.get(abordagem),
            "pergunta": item["pergunta"],
            "gabarito_oficial": item["gabarito"],
            "resposta_gerada": f"ERRO DE INFERENCIA: {str(exc)}",
            "rastros_execucao": {},
            "selecao_gflow": {},
            "numero_chamadas_slm": 3 if abordagem == "gflow" else 1,
            "telemetria": {},
            "status": "erro",
            "duracao_segundos": duracao,
        }


# ==========================================
# 6. FUNCAO PRINCIPAL DE ORQUESTRACAO
# ==========================================
async def main():
    configurar_arquivos_saida()
    configurar_logging()
    logger.info("Preparando experimento v1 com modelo %s.", MODEL_NAME)
    logger.info("Diretorio desta rodada: %s", RUN_DIR)

    resultados_por_chave = carregar_checkpoint() if RETOMANDO else {}
    if not RETOMANDO:
        with open(PARTIAL_OUTPUT_FILE, "w", encoding="utf-8"):
            pass
    salvar_resultados_consolidados(list(resultados_por_chave.values()))
    logger.info(
        "%s: %s resultados recuperados.",
        "Retomando checkpoint" if RETOMANDO else "Arquivos de saida inicializados",
        len(resultados_por_chave),
    )

    dados = carregar_dados_amostra()
    salvar_manifesto(
        RUN_DIR,
        {
            "modelo": MODEL_NAME,
            "temperature": 0.0,
            "top_p": 0.9,
            "task_concurrency_limit": TASK_CONCURRENCY_LIMIT,
            "call_concurrency_limit": CALL_CONCURRENCY_LIMIT,
            "seed": EXPERIMENT_SEED,
            "samples_per_dataset": NUM_AMOSTRAS_POR_DATASET,
            "prompt_version": PROMPT_VERSION,
        },
        [item["id"] for item in dados],
        {"prompts": PROMPTS, "gflow": PROMPTS_GFLOW_AGENTES},
    )
    monitor = MonitorRecursos().iniciar()
    sem_tarefas = asyncio.Semaphore(TASK_CONCURRENCY_LIMIT)
    sem_chamadas = asyncio.Semaphore(CALL_CONCURRENCY_LIMIT)

    tarefas = []
    logger.info("Iniciando inferencias paralelas no modelo alvo (%s).", MODEL_NAME)

    for item in dados:
        for abordagem, prompt_sistema in PROMPTS.items():
            checkpoint = resultados_por_chave.get((item["id"], abordagem))
            if checkpoint and checkpoint.get("status") == "ok":
                continue
            tarefas.append(
                processar_instancia(
                    sem_tarefas,
                    sem_chamadas,
                    item,
                    abordagem,
                    prompt_sistema,
                )
            )

    total_tarefas = len(tarefas)
    total_concluidos = sum(
        item.get("status") == "ok" for item in resultados_por_chave.values()
    )

    try:
        for indice, tarefa in enumerate(asyncio.as_completed(tarefas), start=1):
            resultado = await tarefa
            resultados_por_chave[chave_checkpoint_resultado(resultado)] = resultado
            salvar_resultado_parcial(resultado)
            salvar_resultados_consolidados(list(resultados_por_chave.values()))

            logger.info(
                (
                    "Resultado salvo %s/%s | id=%s | dataset=%s | abordagem=%s | "
                    "status=%s | duracao=%ss\n"
                    "Gabarito esperado:\n%s\n"
                    "Resposta do LLM:\n%s"
                ),
                total_concluidos + indice,
                total_concluidos + total_tarefas,
                resultado["id_instancia"],
                resultado["dataset"],
                resultado["abordagem"],
                resultado["status"],
                resultado["duracao_segundos"],
                texto_para_log(resultado["gabarito_oficial"]),
                texto_para_log(resultado["resposta_gerada"]),
            )
    finally:
        monitor.finalizar(RUN_DIR / "recursos_execucao.json")

    logger.info("Experimento concluido com sucesso.")
    logger.info(
        "Total de execucoes salvas: %s (%s instancias x %s abordagens).",
        len(resultados_por_chave),
        len(dados),
        len(PROMPTS),
    )
    logger.info("JSON consolidado: %s.", OUTPUT_FILE)
    logger.info("JSONL parcial: %s.", PARTIAL_OUTPUT_FILE)
    logger.info("Log completo: %s.", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
