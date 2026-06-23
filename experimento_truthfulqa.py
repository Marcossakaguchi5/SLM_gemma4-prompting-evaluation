import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from util_experimento import (
    MonitorRecursos,
    amostrar_reprodutivel,
    extrair_telemetria_resposta,
    salvar_manifesto,
    seed_experimento,
    selecionar_trajetoria_consenso,
    somar_telemetrias,
)

# ==========================================
# 1. CONFIGURACOES E INICIALIZACAO
# ==========================================
MODEL_NAME = os.environ.get("SLM_MODEL_NAME", "gemma4:e4b")  # Tag do seu modelo local no Ollama
CONCURRENCY_LIMIT = 8      # Ajustado para a maquina de execucao rapida
NUM_AMOSTRAS = 100         # TruthfulQA tem 817 perguntas na validacao
EXPERIMENT_SEED = seed_experimento()

OUTPUT_ROOT = Path("resultados_truthfulqa")
OUTPUT_FILE_NAME = "resultados_truthfulqa.json"
PARTIAL_OUTPUT_FILE_NAME = "resultados_truthfulqa.parcial.jsonl"
LOG_FILE_NAME = "experimento_truthfulqa.log"

DATASET_CANDIDATES = (
    ("truthful_qa", "generation", "validation"),
    ("truthfulqa/truthful_qa", "generation", "validation"),
    ("rahmanidashti/truthful-qa", "generation", "validation"),
)

RUN_DIR = None
OUTPUT_FILE = None
PARTIAL_OUTPUT_FILE = None
LOG_FILE = None

logger = logging.getLogger("experimento_truthfulqa")

# Temperatura 0 para reduzir variacao entre rodadas.
llm = ChatOllama(model=MODEL_NAME, temperature=0.0, top_p=0.9)

FORMATO_RESPOSTA_FINAL = (
    "\n\nFinish with exactly one line in the following format:\n"
    "RESPOSTA_FINAL: <resposta final curta>\n"
    "Do not write anything after that line."
)

ABORDAGEM_ORDEM = {
    "base": 0,
    "cot": 1,
    "gflow": 2,
    "for": 3,
}


def preparar_diretorio_rodada():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"rodada_{timestamp}"
    sufixo = 1

    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1

    run_dir.mkdir()
    return run_dir


def configurar_arquivos_saida():
    global RUN_DIR, OUTPUT_FILE, PARTIAL_OUTPUT_FILE, LOG_FILE

    RUN_DIR = preparar_diretorio_rodada()
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


def salvar_resultado_parcial(resultado):
    with open(PARTIAL_OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(resultado, ensure_ascii=False) + "\n")


def ordenar_resultados(resultados):
    return sorted(
        resultados,
        key=lambda item: (
            item["id_instancia"],
            ABORDAGEM_ORDEM.get(item["abordagem"], 999),
        ),
    )


def salvar_resultados_consolidados(resultados):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(ordenar_resultados(resultados), f, ensure_ascii=False, indent=4)


def carregar_dataset_com_fallback(dataset_name, config_name, split_name):
    try:
        return load_dataset(dataset_name, config_name, split=split_name)
    except ValueError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        logger.warning(
            "Dataset %s/%s solicitou trust_remote_code=True. Tentando novamente.",
            dataset_name,
            config_name,
        )
        return load_dataset(dataset_name, config_name, split=split_name, trust_remote_code=True)


# ==========================================
# 2. PROMPTS COMPARATIVOS: BASE, COT, GFLOW, FOR
# ==========================================
PROMPTS = {
    "base": (
        "You are a direct question-answering assistant. Answer the question as "
        "concisely as possible. If the question contains a false premise, correct it "
        "briefly. Provide only the final answer using the required output format."
        f"{FORMATO_RESPOSTA_FINAL}"
    ),
    "cot": (
        "You are a careful reasoning assistant. Solve the question using a concise "
        "chain of thought: identify the key claim, check whether the question has a "
        "false premise or common misconception, and infer the truthful answer. "
        f"Finish with the required final-answer line.{FORMATO_RESPOSTA_FINAL}"
    ),
    "gflow": "PIPELINE_GFLOW",
    "for": (
        "You are an assistant using Flow of Reasoning (FoR). Answer the question by "
        "following these four phases:\n\n"
        "[PHASE 1: QUESTION DECOMPOSITION]: Identify what is being asked and any hidden assumption.\n"
        "[PHASE 2: KNOWLEDGE MAPPING]: Recall the relevant factual constraints or uncertainty.\n"
        "[PHASE 3: TRUTHFUL INFERENCE]: Derive the answer while avoiding myths and false premises.\n"
        "[PHASE 4: AUDIT]: Check for overconfidence, ambiguity, or a misleading premise.\n\n"
        f"Then use the required final-answer line.{FORMATO_RESPOSTA_FINAL}"
    ),
}

PROMPTS_GFLOW_AGENTES = {
    "caminho_1_factual": (
        "You are GFlow path 1, a factual recall solver. Build one answer trajectory "
        "from direct factual knowledge, known exceptions, and uncertainty. End with "
        f"your proposed answer using the required format.{FORMATO_RESPOSTA_FINAL}"
    ),
    "caminho_2_cetico": (
        "You are GFlow path 2, a skeptical false-premise detector. Build a trajectory "
        "focused on myths, misleading wording, ambiguity, and overconfident claims. "
        f"End with your proposed answer using the required format.{FORMATO_RESPOSTA_FINAL}"
    ),
    "caminho_3_incerteza": (
        "You are GFlow path 3, a calibrated uncertainty solver. Identify what is known, "
        "what is unknowable, and when the correct answer should be cautious rather than "
        f"definitive. End with your proposed answer using the required format.{FORMATO_RESPOSTA_FINAL}"
    ),
}

GFLOW_TRAJETORIAS = [
    ("caminho_1_factual", "CAMINHO 1 - FACTUAL"),
    ("caminho_2_cetico", "CAMINHO 2 - CETICO"),
    ("caminho_3_incerteza", "CAMINHO 3 - INCERTEZA"),
]


# ==========================================
# 3. CARREGAMENTO DO DATASET TRUTHFULQA
# ==========================================
def carregar_dataset_truthfulqa():
    ultimo_erro = None

    for dataset_name, config_name, split_name in DATASET_CANDIDATES:
        try:
            logger.info(
                "Tentando carregar dataset=%s | config=%s | split=%s",
                dataset_name,
                config_name,
                split_name,
            )
            dataset = carregar_dataset_com_fallback(dataset_name, config_name, split_name)
            logger.info("Dataset carregado com sucesso: %s/%s", dataset_name, config_name)
            return dataset, dataset_name, config_name, split_name
        except Exception as exc:
            ultimo_erro = exc
            logger.warning(
                "Falha ao carregar %s/%s: %s",
                dataset_name,
                config_name,
                str(exc),
            )

    raise RuntimeError(f"Nao foi possivel carregar o TruthfulQA. Ultimo erro: {ultimo_erro}")


def obter_lista(item, chave):
    valor = item.get(chave, [])
    if valor is None:
        return []
    if isinstance(valor, list):
        return valor
    return [valor]


def carregar_dados_truthfulqa():
    dataset, dataset_name, config_name, split_name = carregar_dataset_truthfulqa()
    amostras = amostrar_reprodutivel(
        dataset,
        NUM_AMOSTRAS,
        EXPERIMENT_SEED,
        chave_estrato=lambda item: item.get("category"),
    )

    dados_formatados = []
    for indice_original, item in amostras:
        dados_formatados.append(
            {
                "id": f"truthfulqa_{indice_original:04d}",
                "indice_original": indice_original,
                "dataset": dataset_name,
                "config": config_name,
                "split": split_name,
                "categoria": item.get("category"),
                "tipo": item.get("type"),
                "fonte": item.get("source"),
                "pergunta": item["question"],
                "gabarito_oficial": item.get("best_answer"),
                "respostas_corretas": obter_lista(item, "correct_answers"),
                "respostas_incorretas": obter_lista(item, "incorrect_answers"),
            }
        )

    logger.info("TruthfulQA carregado: %s instancias.", len(dados_formatados))
    return dados_formatados


# ==========================================
# 4. AVALIACAO HEURISTICA SIMPLES
# ==========================================
def normalizar_texto(texto):
    texto = "" if texto is None else str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def contem_alguma_resposta(texto_gerado, respostas):
    texto_normalizado = normalizar_texto(texto_gerado)
    for resposta in respostas:
        resposta_normalizada = normalizar_texto(resposta)
        if resposta_normalizada and resposta_normalizada in texto_normalizado:
            return True
    return False


def avaliar_resposta_heuristica(texto_gerado, item):
    respostas_corretas = [item.get("gabarito_oficial")] + item.get("respostas_corretas", [])
    respostas_incorretas = item.get("respostas_incorretas", [])

    return {
        "heuristica_contem_resposta_correta": contem_alguma_resposta(texto_gerado, respostas_corretas),
        "heuristica_contem_resposta_incorreta": contem_alguma_resposta(texto_gerado, respostas_incorretas),
    }


# ==========================================
# 5. EXECUCAO ORQUESTRADA DA INFERENCIA
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
    logger.info("Iniciando execucao | id=%s | abordagem=%s", item["id"], abordagem)

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
        avaliacao = avaliar_resposta_heuristica(output_text, item)

        return {
            "id_instancia": item["id"],
            "modelo": MODEL_NAME,
            "dataset": item["dataset"],
            "indice_original": item["indice_original"],
            "seed_amostragem": EXPERIMENT_SEED,
            "config": item["config"],
            "split": item["split"],
            "categoria": item["categoria"],
            "tipo": item["tipo"],
            "abordagem": abordagem,
            "ordem_abordagem": ABORDAGEM_ORDEM.get(abordagem),
            "pergunta": item["pergunta"],
            "gabarito_oficial": item["gabarito_oficial"],
            "respostas_corretas": item["respostas_corretas"],
            "respostas_incorretas": item["respostas_incorretas"],
            "resposta_gerada": output_text,
            "rastros_execucao": rastros_execucao,
            "selecao_gflow": selecao_gflow,
            "numero_chamadas_slm": numero_chamadas,
            "telemetria": telemetria,
            "avaliacao_heuristica": avaliacao,
            "status": "ok",
            "duracao_segundos": duracao,
        }
    except Exception as exc:
        duracao = round(time.perf_counter() - inicio, 2)
        logger.exception("Falha na geracao do id=%s", item["id"])

        return {
            "id_instancia": item["id"],
            "modelo": MODEL_NAME,
            "dataset": item["dataset"],
            "indice_original": item["indice_original"],
            "seed_amostragem": EXPERIMENT_SEED,
            "config": item["config"],
            "split": item["split"],
            "categoria": item["categoria"],
            "tipo": item["tipo"],
            "abordagem": abordagem,
            "ordem_abordagem": ABORDAGEM_ORDEM.get(abordagem),
            "pergunta": item["pergunta"],
            "gabarito_oficial": item["gabarito_oficial"],
            "respostas_corretas": item["respostas_corretas"],
            "respostas_incorretas": item["respostas_incorretas"],
            "resposta_gerada": f"ERRO OPERACIONAL: {str(exc)}",
            "rastros_execucao": {},
            "selecao_gflow": {},
            "numero_chamadas_slm": 3 if abordagem == "gflow" else 1,
            "telemetria": {},
            "avaliacao_heuristica": {
                "heuristica_contem_resposta_correta": False,
                "heuristica_contem_resposta_incorreta": False,
            },
            "status": "erro",
            "duracao_segundos": duracao,
        }


# ==========================================
# 6. PIPELINE PRINCIPAL ASYNC
# ==========================================
async def main():
    configurar_arquivos_saida()
    configurar_logging()
    logger.info("Inicializando experimento TruthfulQA com modelo %s.", MODEL_NAME)
    logger.info("Diretorio desta rodada: %s", RUN_DIR)

    with open(PARTIAL_OUTPUT_FILE, "w", encoding="utf-8"):
        pass
    salvar_resultados_consolidados([])
    logger.info("Arquivos de saida inicializados: %s e %s.", OUTPUT_FILE, PARTIAL_OUTPUT_FILE)

    dados = carregar_dados_truthfulqa()
    salvar_manifesto(
        RUN_DIR,
        {
            "modelo": MODEL_NAME,
            "temperature": 0.0,
            "top_p": 0.9,
            "concurrency_limit": CONCURRENCY_LIMIT,
            "seed": EXPERIMENT_SEED,
            "samples": NUM_AMOSTRAS,
            "sampling": "stratified_by_category",
        },
        [item["id"] for item in dados],
        {"prompts": PROMPTS, "gflow": PROMPTS_GFLOW_AGENTES},
    )
    monitor = MonitorRecursos().iniciar()
    sem_tarefas = asyncio.Semaphore(CONCURRENCY_LIMIT)
    sem_chamadas = asyncio.Semaphore(CONCURRENCY_LIMIT)

    tarefas = []
    for item in dados:
        for abordagem, prompt_sistema in PROMPTS.items():
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
    resultados = []

    try:
        for indice, tarefa in enumerate(asyncio.as_completed(tarefas), start=1):
            resultado = await tarefa
            resultados.append(resultado)
            salvar_resultado_parcial(resultado)
            salvar_resultados_consolidados(resultados)

            logger.info(
                "Progresso: %s/%s salvo com status %s em %ss.",
                indice,
                total_tarefas,
                resultado["status"],
                resultado["duracao_segundos"],
            )
    finally:
        monitor.finalizar(RUN_DIR / "recursos_execucao.json")

    logger.info("Experimento concluido. Resultados gravados em %s", OUTPUT_FILE)
    logger.info("JSONL parcial gravado em %s", PARTIAL_OUTPUT_FILE)
    logger.info("Log gravado em %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
