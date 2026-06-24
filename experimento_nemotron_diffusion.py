"""Executa o Nemotron Diffusion sobre a mesma amostra ja usada pelo Gemma.

Este arquivo nao baixa nem reamostra datasets. Ele recebe os diretorios brutos
da geracao principal, reaproveita pergunta, gabarito e identificador de cada
instancia e gera uma resposta-base do Nemotron para cada uma delas.
"""

import argparse
import json
import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from configuracao.ambiente import decimal, inteiro, texto
from configuracao.prompts import (
    PROMPT_VERSION,
    PROMPTS_GSM8K_ARC,
    PROMPTS_HENDRYCKS_MATH,
    PROMPTS_MATH_AVANCADO,
    PROMPTS_TRUTHFULQA,
)
MODELO_PADRAO = "nvidia/Nemotron-Labs-Diffusion-3B"
MODOS_VALIDOS = ("diffusion", "ar", "linear-spec")


MODELO_NEMOTRON = texto("NEMOTRON_MODEL_ID", MODELO_PADRAO)
MODO = texto("NEMOTRON_MODE", "diffusion")
MAX_NEW_TOKENS = inteiro("NEMOTRON_MAX_NEW_TOKENS", 128, minimo=1)
BLOCK_LENGTH = inteiro("NEMOTRON_BLOCK_LENGTH", 32, minimo=1)
THRESHOLD = decimal("NEMOTRON_THRESHOLD", 0.9, minimo=0, maximo=1)
TEMPERATURE = decimal("NEMOTRON_TEMPERATURE", 0.0, minimo=0)
DEVICE = texto("NEMOTRON_DEVICE", "auto")
DTYPE = texto("NEMOTRON_DTYPE", "auto")
OUTPUT_ROOT = Path(texto("EXPERIMENT_OUTPUT_ROOT", "resultados_nemotron_diffusion"))

OUTPUT_FILE_NAME = "resultados_nemotron_diffusion.json"
PARTIAL_OUTPUT_FILE_NAME = "resultados_nemotron_diffusion.parcial.jsonl"
LOG_FILE_NAME = "experimento_nemotron_diffusion.log"
ABORDAGEM = "base"


def preparar_diretorio_rodada(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"rodada_{timestamp}"
    sufixo = 1
    while run_dir.exists():
        run_dir = output_root / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1
    run_dir.mkdir()
    return run_dir


def configurar_logging(log_file):
    logger = logging.getLogger("experimento_nemotron_diffusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def descobrir_arquivos_resultado(fontes):
    arquivos = []
    for fonte_bruta in fontes:
        fonte = Path(fonte_bruta)
        if fonte.is_dir():
            candidatos = fonte.rglob("resultados_*.json")
        else:
            candidatos = (fonte,)
        for arquivo in candidatos:
            nome = arquivo.name.lower()
            if (
                arquivo.suffix.lower() == ".json"
                and not nome.endswith(".parcial.json")
                and "avaliacao" not in nome
                and "manifesto" not in nome
                and "recursos" not in nome
            ):
                arquivos.append(arquivo)
    return sorted(set(arquivos))


def carregar_lista_resultados(arquivo):
    with arquivo.open("r", encoding="utf-8") as entrada:
        dados = json.load(entrada)
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict) and isinstance(dados.get("resultados"), list):
        return dados["resultados"]
    raise ValueError(f"{arquivo} nao contem uma lista de resultados.")


def carregar_referencias(caminhos):
    referencias = {}
    for caminho_bruto in caminhos or []:
        caminho = Path(caminho_bruto)
        if not caminho.is_file():
            raise FileNotFoundError(f"Arquivo de referencias nao encontrado: {caminho}")
        for item in carregar_lista_resultados(caminho):
            if not isinstance(item, dict) or not item.get("id_instancia"):
                continue
            chave = (item["id_instancia"], item.get("dataset", ""))
            anterior = referencias.get(chave)
            if anterior and anterior != item:
                raise ValueError(
                    "A mesma instancia apareceu com referencias diferentes: "
                    f"{chave[0]} / {chave[1]}."
                )
            referencias[chave] = dict(item)
    return referencias


def extrair_instancias_canonicas(fontes, referencias=None):
    """Seleciona perguntas canonicas e, opcionalmente, anexa suas referencias."""
    instancias = {}
    arquivos = descobrir_arquivos_resultado(fontes)
    if not arquivos:
        raise FileNotFoundError("Nenhum JSON bruto de resultados foi encontrado nas fontes.")

    for arquivo in arquivos:
        for item in carregar_lista_resultados(arquivo):
            if not isinstance(item, dict):
                continue
            if not item.get("id_instancia") or not item.get("pergunta"):
                continue
            chave = (item["id_instancia"], item.get("dataset", ""))
            anterior = instancias.get(chave)
            if anterior and anterior.get("pergunta") != item.get("pergunta"):
                raise ValueError(
                    "A mesma instancia apareceu com perguntas diferentes: "
                    f"{chave[0]} / {chave[1]}."
                )
            if anterior is None:
                copia = dict(item)
                copia["_arquivo_origem_nemotron"] = str(arquivo)
                instancias[chave] = copia

    if not instancias:
        raise RuntimeError("As fontes nao contem instancias validas para o Nemotron.")

    referencias_por_id = carregar_referencias(referencias)
    if referencias:
        faltantes = []
        for chave, item in instancias.items():
            referencia = referencias_por_id.get(chave)
            if referencia is None:
                faltantes.append(f"{chave[0]} / {chave[1]}")
                continue
            for campo, valor in referencia.items():
                if campo not in {"pergunta", "resposta_gerada", "abordagem", "modelo"}:
                    item[campo] = valor
        if faltantes:
            raise RuntimeError(
                "Faltam referencias para as perguntas: " + ", ".join(faltantes[:5])
            )

    return sorted(
        instancias.values(),
        key=lambda item: (str(item.get("dataset", "")), str(item["id_instancia"])),
    )


def validar_tamanho_amostra(instancias, esperado):
    if esperado is None:
        return
    contagens = Counter(item.get("dataset", "") for item in instancias)
    diferentes = {
        dataset: quantidade
        for dataset, quantidade in contagens.items()
        if quantidade != esperado
    }
    if diferentes:
        detalhe = ", ".join(
            f"{dataset}={quantidade}" for dataset, quantidade in sorted(diferentes.items())
        )
        raise RuntimeError(
            "A amostra de origem nao tem o tamanho esperado por dataset "
            f"({esperado}): {detalhe}."
        )


def prompt_base_para(item):
    dataset = str(item.get("dataset", "")).lower()
    if "truthful" in dataset:
        return PROMPTS_TRUTHFULQA["base"]
    if "hendrycks" in dataset or "math" in dataset:
        return PROMPTS_HENDRYCKS_MATH["base"]
    if "gsm8k" in dataset or "arc" in dataset:
        return PROMPTS_GSM8K_ARC["base"]
    return PROMPTS_MATH_AVANCADO["base"]


def campos_contextuais(item):
    nomes = (
        "indice_original",
        "seed_amostragem",
        "subset",
        "level",
        "config",
        "split",
        "categoria",
        "tipo",
        "fonte",
        "gabarito_oficial",
        "gabarito",
        "resposta_boxed",
        "respostas_corretas",
        "respostas_incorretas",
    )
    return {nome: item[nome] for nome in nomes if nome in item}


def montar_registro(item, resposta, telemetria, duracao, status="ok", erro=None):
    registro = {
        "id_instancia": item["id_instancia"],
        "modelo": MODELO_NEMOTRON,
        "familia_modelo": "nemotron_diffusion",
        "tipo_condicao": "modelo_baseline_externo",
        "dataset": item.get("dataset"),
        "abordagem": ABORDAGEM,
        "ordem_abordagem": 0,
        "pergunta": item.get("pergunta"),
        "resposta_gerada": resposta,
        "rastros_execucao": {},
        "selecao_gflow": {},
        "numero_chamadas_slm": 1,
        "telemetria": telemetria,
        "status": status,
        "duracao_segundos": round(duracao, 2),
        "configuracao_nemotron": {
            "model_id": MODELO_NEMOTRON,
            "mode": MODO,
            "max_new_tokens": MAX_NEW_TOKENS,
            "block_length": BLOCK_LENGTH,
            "threshold": THRESHOLD,
            "temperature": TEMPERATURE,
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "arquivo_origem_amostra": item.get("_arquivo_origem_nemotron"),
    }
    registro.update(campos_contextuais(item))
    if erro:
        registro["erro"] = str(erro)
    return registro


def salvar_json(caminho, dados):
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")


def salvar_manifesto(run_dir, fontes, referencias, instancias):
    manifesto = {
        "tipo": "experimento_modelo_baseline_externo",
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo": MODELO_NEMOTRON,
        "familia_modelo": "nemotron_diffusion",
        "abordagem": ABORDAGEM,
        "prompt_version": PROMPT_VERSION,
        "parametros_geracao": {
            "mode": MODO,
            "max_new_tokens": MAX_NEW_TOKENS,
            "block_length": BLOCK_LENGTH,
            "threshold": THRESHOLD,
            "temperature": TEMPERATURE,
            "device": DEVICE,
            "dtype": DTYPE,
        },
        "fontes_amostra": [str(Path(fonte)) for fonte in fontes],
        "fontes_referencias": [str(Path(fonte)) for fonte in referencias or []],
        "ids_amostra": [item["id_instancia"] for item in instancias],
        "contagem_por_dataset": dict(
            sorted(Counter(item.get("dataset", "") for item in instancias).items())
        ),
    }
    salvar_json(run_dir / "manifesto_execucao.json", manifesto)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Executa o Nemotron Diffusion sobre os mesmos itens ja gerados pelos "
            "benchmarks principais."
        )
    )
    parser.add_argument(
        "fontes",
        nargs="+",
        help="Diretorios/JSONs com as perguntas ou resultados brutos da geracao.",
    )
    parser.add_argument(
        "--referencias",
        nargs="+",
        default=[],
        help="JSON separado com gabaritos e metadados para avaliacao.",
    )
    parser.add_argument(
        "--expected-per-dataset",
        type=int,
        default=None,
        help="Falha se uma fonte nao tiver esta quantidade por dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if MODO not in MODOS_VALIDOS:
        raise ValueError(f"NEMOTRON_MODE deve ser um de {MODOS_VALIDOS}.")
    if args.expected_per_dataset is not None and args.expected_per_dataset < 1:
        raise ValueError("--expected-per-dataset deve ser maior ou igual a 1.")

    instancias = extrair_instancias_canonicas(args.fontes, args.referencias)
    validar_tamanho_amostra(instancias, args.expected_per_dataset)
    try:
        from modelo_nemotron_diffusion import NemotronDiffusionRunner
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "transformers"}:
            raise RuntimeError(
                "Nemotron requer torch e transformers. Ative o ambiente virtual e "
                "execute: pip install -r requirements.txt"
            ) from exc
        raise
    run_dir = preparar_diretorio_rodada(OUTPUT_ROOT)
    output_file = run_dir / OUTPUT_FILE_NAME
    partial_file = run_dir / PARTIAL_OUTPUT_FILE_NAME
    logger = configurar_logging(run_dir / LOG_FILE_NAME)
    salvar_manifesto(run_dir, args.fontes, args.referencias, instancias)
    salvar_json(output_file, [])

    logger.info("Diretorio da rodada Nemotron: %s", run_dir)
    logger.info("Itens recebidos: %s", len(instancias))
    logger.info("Carregando o modelo uma unica vez: %s", MODELO_NEMOTRON)
    inicio_carregamento = time.perf_counter()
    runner = NemotronDiffusionRunner(
        model_id=MODELO_NEMOTRON,
        modo=MODO,
        max_new_tokens=MAX_NEW_TOKENS,
        block_length=BLOCK_LENGTH,
        threshold=THRESHOLD,
        temperature=TEMPERATURE,
        device=DEVICE,
        dtype=DTYPE,
    )
    logger.info(
        "Modelo carregado em %.2fs no dispositivo %s.",
        time.perf_counter() - inicio_carregamento,
        runner.torch_device,
    )

    resultados = []
    with partial_file.open("w", encoding="utf-8") as parcial:
        for indice, item in enumerate(instancias, start=1):
            inicio = time.perf_counter()
            try:
                resposta, telemetria = runner.gerar(
                    item["pergunta"],
                    instrucao_sistema=prompt_base_para(item),
                )
                registro = montar_registro(
                    item,
                    resposta,
                    telemetria,
                    time.perf_counter() - inicio,
                )
            except Exception as exc:
                logger.exception("Falha no item %s.", item["id_instancia"])
                registro = montar_registro(
                    item,
                    f"ERRO DE INFERENCIA: {exc}",
                    {},
                    time.perf_counter() - inicio,
                    status="erro",
                    erro=exc,
                )
            resultados.append(registro)
            parcial.write(json.dumps(registro, ensure_ascii=False) + "\n")
            parcial.flush()
            salvar_json(output_file, resultados)
            logger.info(
                "[%s/%s] %s | %s | %s | %.2fs",
                indice,
                len(instancias),
                registro.get("dataset"),
                registro.get("id_instancia"),
                registro.get("status"),
                registro.get("duracao_segundos"),
            )

    logger.info("Concluido: %s respostas salvas em %s.", len(resultados), output_file)


if __name__ == "__main__":
    main()
