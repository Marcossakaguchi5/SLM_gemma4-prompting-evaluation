import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from configuracao.ambiente import inteiro, texto
from metricas_avaliacao import extrair_resposta_boxed
from configuracao.prompts import (
    PROMPT_VERSION,
    PROMPTS_GFLOW_MATH_AVANCADO as PROMPTS_GFLOW_AGENTES,
    PROMPTS_MATH_AVANCADO as PROMPTS,
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
# 1. CONFIGURAÇÕES E INICIALIZAÇÃO
# ==========================================
MODEL_NAME = texto("SLM_MODEL_NAME", "gemma4:e4b")
TASK_CONCURRENCY_LIMIT = inteiro("EXPERIMENT_TASK_CONCURRENCY", 16, minimo=1)
CALL_CONCURRENCY_LIMIT = inteiro("EXPERIMENT_CALL_CONCURRENCY", 4, minimo=1)
NUM_AMOSTRAS = inteiro("EXPERIMENT_NUM_SAMPLES", 100, minimo=1)
EXPERIMENT_SEED = seed_experimento()
OUTPUT_ROOT = Path(texto("EXPERIMENT_OUTPUT_ROOT", "resultados_math_avancado"))
OUTPUT_FILE_NAME = "resultados_math_avancado.json"
PARTIAL_OUTPUT_FILE_NAME = "resultados_math_avancado.parcial.jsonl"
LOG_FILE_NAME = "experimento_math_avancado.log"
HENDRYCKS_MATH_DATASET = "EleutherAI/hendrycks_math"
HENDRYCKS_MATH_SUBSETS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

RUN_DIR = None
OUTPUT_FILE = None
PARTIAL_OUTPUT_FILE = None
LOG_FILE = None
RETOMANDO = False

logger = logging.getLogger("experimento_math_avancado")

# Inicialização com ganância estrita (temperatura 0) para reprodutibilidade científica
llm = ChatOllama(model=MODEL_NAME, temperature=0.0, top_p=0.9)

ABORDAGEM_ORDEM = {
    "base": 0,
    "cot": 1,
    "gflow": 2,
    "for": 3,
}

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
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def salvar_resultado_parcial(res):
    adicionar_jsonl_duravel(PARTIAL_OUTPUT_FILE, res)

def salvar_resultados_consolidados(res_lista):
    salvar_json_atomico(OUTPUT_FILE, res_lista, indent=4)


def carregar_checkpoint():
    return carregar_checkpoint_jsonl(PARTIAL_OUTPUT_FILE, chave_checkpoint_resultado)

GFLOW_TRAJETORIAS = [
    ("caminho_1_formal", "CAMINHO 1 - FORMAL"),
    ("caminho_2_heuristico", "CAMINHO 2 - HEURISTICO"),
    ("caminho_3_casos", "CAMINHO 3 - CASOS"),
]

# ==========================================
# 3. CARREGAMENTO DO DATASET COMPLEXO (AIME)
# ==========================================
def carregar_dados_aime():
    logger.info("Carregando o dataset de alta complexidade (Competition Math - Nível AIME)...")

    dataset = []
    for subset in HENDRYCKS_MATH_SUBSETS:
        logger.info("Carregando subconjunto %s/%s...", HENDRYCKS_MATH_DATASET, subset)
        dataset.extend(load_dataset(HENDRYCKS_MATH_DATASET, subset, split="test"))

    # Filtrando instâncias tipificadas como 'aime' ou de alto nível (Level 5)
    questoes_filtradas = [q for q in dataset if q.get("level") == "Level 5" or "aime" in str(q.get("type")).lower()]

    if not questoes_filtradas:
        raise RuntimeError("Nenhuma questão Level 5/AIME foi encontrada no dataset carregado.")

    amostras = amostrar_reprodutivel(
        questoes_filtradas,
        NUM_AMOSTRAS,
        EXPERIMENT_SEED,
        chave_estrato=lambda item: item.get("type"),
    )

    dados_formatados = []
    for indice_original, item in amostras:
        dados_formatados.append({
            "id": f"math_advanced_{indice_original:04d}",
            "indice_original": indice_original,
            "dataset": HENDRYCKS_MATH_DATASET,
            "subset": item.get("type"),
            "level": item.get("level"),
            "pergunta": item["problem"],
            "gabarito": item["solution"],
            "resposta_boxed": extrair_resposta_boxed(item["solution"]),
        })

    logger.info("Dataset de Competição carregado com sucesso: %s instâncias de nível olímpico.", len(dados_formatados))
    return dados_formatados

# ==========================================
# 4. EXECUCAO ORQUESTRADA DA INFERENCIA
# ==========================================
async def executar_trajetoria_gflow(sem_chamadas, chave_prompt, pergunta):
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", PROMPTS_GFLOW_AGENTES[chave_prompt]),
            ("human", "{input}")
        ])
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
    logger.info("Iniciando execução | id=%s | abordagem=%s", item["id"], abordagem)
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
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", prompt_sistema),
                ("human", "{input}")
            ])
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
            "subset": item["subset"],
            "level": item["level"],
            "abordagem": abordagem,
            "ordem_abordagem": ABORDAGEM_ORDEM.get(abordagem),
            "pergunta": item["pergunta"],
            "gabarito_oficial": item["gabarito"],
            "resposta_boxed": item["resposta_boxed"],
            "resposta_gerada": output_text,
            "rastros_execucao": rastros_execucao,
            "selecao_gflow": selecao_gflow,
            "numero_chamadas_slm": numero_chamadas,
            "telemetria": telemetria,
            "status": "ok",
            "duracao_segundos": duracao
        }
    except Exception as e:
        duracao = round(time.perf_counter() - inicio, 2)
        logger.error("Falha crítica na geração do id=%s: %s", item["id"], str(e))
        return {
            "id_instancia": item["id"],
            "modelo": MODEL_NAME,
            "dataset": item["dataset"],
            "indice_original": item["indice_original"],
            "seed_amostragem": EXPERIMENT_SEED,
            "subset": item["subset"],
            "level": item["level"],
            "abordagem": abordagem,
            "ordem_abordagem": ABORDAGEM_ORDEM.get(abordagem),
            "pergunta": item["pergunta"],
            "gabarito_oficial": item["gabarito"],
            "resposta_boxed": item["resposta_boxed"],
            "resposta_gerada": f"ERRO OPERACIONAL: {str(e)}",
            "rastros_execucao": {},
            "selecao_gflow": {},
            "numero_chamadas_slm": 3 if abordagem == "gflow" else 1,
            "telemetria": {},
            "status": "erro",
            "duracao_segundos": duracao
        }

# ==========================================
# 5. PIPELINE PRINCIPAL ASYNC
# ==========================================
async def main():
    configurar_arquivos_saida()
    configurar_logging()
    logger.info("Inicializando pipeline experimental avançado com modelo %s.", MODEL_NAME)
    logger.info("Diretório desta rodada: %s", RUN_DIR)

    # Inicializa os arquivos da rodada atual sem tocar nas rodadas anteriores.
    resultados_por_chave = carregar_checkpoint() if RETOMANDO else {}
    if not RETOMANDO:
        with open(PARTIAL_OUTPUT_FILE, "w", encoding="utf-8"):
            pass
    salvar_resultados_consolidados(list(resultados_por_chave.values()))
    logger.info("Arquivos de saída inicializados: %s e %s.", OUTPUT_FILE, PARTIAL_OUTPUT_FILE)

    dados = carregar_dados_aime()
    salvar_manifesto(
        RUN_DIR,
        {
            "modelo": MODEL_NAME,
            "temperature": 0.0,
            "top_p": 0.9,
            "task_concurrency_limit": TASK_CONCURRENCY_LIMIT,
            "call_concurrency_limit": CALL_CONCURRENCY_LIMIT,
            "seed": EXPERIMENT_SEED,
            "samples": NUM_AMOSTRAS,
            "prompt_version": PROMPT_VERSION,
            "sampling": "stratified_by_type",
            "prompt_variant": "advanced",
        },
        [item["id"] for item in dados],
        {"prompts": PROMPTS, "gflow": PROMPTS_GFLOW_AGENTES},
    )
    monitor = MonitorRecursos().iniciar()
    sem_tarefas = asyncio.Semaphore(TASK_CONCURRENCY_LIMIT)
    sem_chamadas = asyncio.Semaphore(CALL_CONCURRENCY_LIMIT)

    tarefas = []
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
                "Progresso: %s/%s salvo com status %s em %ss.",
                total_concluidos + indice,
                total_concluidos + total_tarefas,
                resultado["status"],
                resultado["duracao_segundos"],
            )
    finally:
        monitor.finalizar(RUN_DIR / "recursos_execucao.json")

    logger.info("Pipeline executado com sucesso total. Resultados gravados em %s", OUTPUT_FILE)

if __name__ == "__main__":
    asyncio.run(main())
