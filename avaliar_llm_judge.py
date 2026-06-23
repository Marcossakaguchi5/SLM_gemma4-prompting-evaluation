import argparse
import csv
import json
import os
import random
import time
import math
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from configuracao.ambiente import texto
from configuracao.prompts import INSTRUCAO_JULGAMENTO_COMPARATIVO, PROMPT_SISTEMA_JUIZ
from util_experimento import (
    extrair_resposta_final,
    extrair_telemetria_resposta,
    hash_estavel,
)

OUTPUT_ROOT = Path("avaliacoes_llm_judge")

def preparar_diretorio_saida(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"rodada_{timestamp}"
    sufixo = 1

    while run_dir.exists():
        run_dir = output_root / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1

    run_dir.mkdir()
    return run_dir


def carregar_json(path):
    with path.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict) and isinstance(dados.get("resultados"), list):
        return dados["resultados"]
    raise ValueError(f"Arquivo nao contem uma lista de resultados: {path}")


def descobrir_arquivos(caminhos):
    arquivos = []
    for caminho_bruto in caminhos:
        caminho = Path(caminho_bruto)
        if caminho.is_dir():
            arquivos.extend(
                sorted(
                    path
                    for path in caminho.rglob("*.json")
                    if "avaliacao" not in path.name
                    and "metricas" not in path.name
                    and "resumo" not in path.name
                    and "manifesto" not in path.name
                    and "recursos_execucao" not in path.name
                )
            )
        else:
            arquivos.append(caminho)
    return arquivos


def carregar_resultados(caminhos):
    resultados = []
    for arquivo in descobrir_arquivos(caminhos):
        for item in carregar_json(arquivo):
            item = dict(item)
            item["_arquivo_origem"] = str(arquivo)
            resultados.append(item)
    return resultados


def normalizar_abordagem(abordagem):
    abordagem = (abordagem or "").strip()
    chave = abordagem.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "baseline": "base",
        "linha_de_base": "base",
        "resposta_direta": "base",
        "direct": "base",
        "direct_answer": "base",
        "chain_of_thought": "cot",
        "chainofthought": "cot",
        "flow_of_reasoning": "for",
        "gflownet": "gflow",
        "gflownet_inspired": "gflow",
        "gflow_inspired": "gflow",
    }
    if chave in aliases:
        return aliases[chave]
    if chave in {"base", "cot", "for", "gflow"}:
        return chave
    if abordagem == "baseline":
        return "base"
    return abordagem or ""


def agrupar_por_instancia(resultados):
    grupos = {}
    for item in resultados:
        if "resposta_gerada" not in item or "abordagem" not in item:
            continue
        chave = (
            item.get("id_instancia"),
            item.get("dataset"),
            item.get("modelo", ""),
        )
        grupo = grupos.setdefault(
            chave,
            {
                "id_instancia": item.get("id_instancia"),
                "dataset": item.get("dataset"),
                "modelo": item.get("modelo", ""),
                "pergunta": item.get("pergunta"),
                "gabarito_oficial": item.get("gabarito_oficial", item.get("gabarito")),
                "resposta_boxed": item.get("resposta_boxed"),
                "respostas_corretas": item.get("respostas_corretas", []),
                "respostas_incorretas": item.get("respostas_incorretas", []),
                "itens": {},
            },
        )
        abordagem = normalizar_abordagem(item.get("abordagem"))
        grupo["itens"][abordagem] = item
    return list(grupos.values())


def extrair_gsm8k_objetivo(gabarito):
    texto = "" if gabarito is None else str(gabarito)
    if "####" in texto:
        return texto.split("####")[-1].strip()
    return texto.strip()


def obter_referencia_curta(item):
    if item.get("resposta_boxed"):
        return str(item["resposta_boxed"]).strip()

    gabarito = item.get("gabarito_oficial", item.get("gabarito", ""))
    dataset = str(item.get("dataset", "")).lower()

    if "gsm8k" in dataset:
        return extrair_gsm8k_objetivo(gabarito)

    return "" if gabarito is None else str(gabarito).strip()


def montar_payload_julgamento(item):
    return {
        "id_instancia": item.get("id_instancia"),
        "modelo": item.get("modelo"),
        "dataset": item.get("dataset"),
        "abordagem": item.get("abordagem"),
        "pergunta": item.get("pergunta"),
        "gabarito_oficial": item.get("gabarito_oficial", item.get("gabarito")),
        "resposta_esperada_curta": obter_referencia_curta(item),
        "resposta_boxed": item.get("resposta_boxed"),
        "respostas_corretas": item.get("respostas_corretas", []),
        "respostas_incorretas": item.get("respostas_incorretas", []),
        "resposta_final_extraida": extrair_resposta_final(item.get("resposta_gerada", "")),
        "resposta_completa_modelo": item.get("resposta_gerada", ""),
    }


def montar_payload_julgamento_comparativo(
    grupo,
    embaralhar=True,
    ordem_abordagens=None,
    rng=None,
):
    itens = list(grupo["itens"].items())
    if ordem_abordagens:
        posicoes = {
            abordagem: indice
            for indice, abordagem in enumerate(ordem_abordagens)
        }
        itens.sort(key=lambda item: posicoes.get(item[0], len(posicoes)))
    elif embaralhar:
        (rng or random).shuffle(itens)

    item_referencia = next(iter(grupo["itens"].values()))
    respostas = []
    for abordagem, item in itens:
        respostas.append(
            {
                "abordagem": abordagem,
                "status": item.get("status"),
                "duracao_segundos": item.get("duracao_segundos"),
                "resposta_final_extraida": extrair_resposta_final(item.get("resposta_gerada", "")),
                "resposta_completa_modelo": item.get("resposta_gerada", ""),
                "rastros_execucao": item.get("rastros_execucao", {}),
                "selecao_gflow": item.get("selecao_gflow", {}),
            }
        )

    return {
        "id_instancia": grupo.get("id_instancia"),
        "modelo": grupo.get("modelo"),
        "dataset": grupo.get("dataset"),
        "pergunta": grupo.get("pergunta"),
        "gabarito_oficial": grupo.get("gabarito_oficial"),
        "resposta_esperada_curta": obter_referencia_curta(item_referencia),
        "resposta_boxed": grupo.get("resposta_boxed"),
        "respostas_corretas": grupo.get("respostas_corretas", []),
        "respostas_incorretas": grupo.get("respostas_incorretas", []),
        "ordem_apresentada": [resposta["abordagem"] for resposta in respostas],
        "respostas_por_abordagem": respostas,
    }


def extrair_json(texto):
    texto = texto.strip()
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    inicio = texto.find("{")
    fim = texto.rfind("}")
    if inicio != -1 and fim != -1 and fim > inicio:
        return json.loads(texto[inicio : fim + 1])

    raise ValueError("Resposta do juiz nao contem JSON valido.")


def chamar_ollama(prompt, modelo):
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "Instale langchain-core e langchain-ollama para usar --provider ollama."
        ) from exc

    juiz = ChatOllama(model=modelo, temperature=0.0, top_p=0.9)
    resposta = juiz.invoke(
        [
            SystemMessage(content=PROMPT_SISTEMA_JUIZ),
            HumanMessage(content=prompt),
        ]
    )
    return resposta.content, extrair_telemetria_resposta(resposta)


def chamar_gemini(prompt, modelo):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Defina GEMINI_API_KEY ou GOOGLE_API_KEY no ambiente. Este script nao carrega .env."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Instale google-genai para usar --provider gemini.") from exc

    client = genai.Client(api_key=api_key)
    resposta = client.models.generate_content(
        model=modelo,
        contents=f"{PROMPT_SISTEMA_JUIZ}\n\n{prompt}",
        config={"response_mime_type": "application/json"},
    )
    usage = getattr(resposta, "usage_metadata", None)
    telemetria = {
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }
    return resposta.text, telemetria


def chamar_openrouter(prompt, modelo):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Defina OPENROUTER_API_KEY no arquivo .env.")

    url = os.environ.get(
        "OPENROUTER_API_URL",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    app_url = os.environ.get("OPENROUTER_APP_URL")
    app_title = os.environ.get("OPENROUTER_APP_TITLE")
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-Title"] = app_title

    corpo = json.dumps(
        {
            "model": modelo,
            "messages": [
                {"role": "system", "content": PROMPT_SISTEMA_JUIZ},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 0.9,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    requisicao = urllib.request.Request(url, data=corpo, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(requisicao, timeout=120) as resposta:
            dados = json.loads(resposta.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detalhe = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"OpenRouter retornou HTTP {exc.code}: {detalhe}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Falha de conexao com o OpenRouter: {exc.reason}") from exc

    try:
        texto = dados["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Resposta do OpenRouter nao contem choices[0].message.content.") from exc
    usage = dados.get("usage") or {}
    telemetria = {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }
    return texto, telemetria


def julgar_grupo(
    grupo,
    provider,
    judge_model,
    embaralhar=True,
    ordem_abordagens=None,
    rng=None,
):
    payload = montar_payload_julgamento_comparativo(
        grupo,
        embaralhar=embaralhar,
        ordem_abordagens=ordem_abordagens,
        rng=rng,
    )
    prompt = (
        f"{INSTRUCAO_JULGAMENTO_COMPARATIVO}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    inicio = time.perf_counter()
    if provider == "gemini":
        texto_julgamento, telemetria = chamar_gemini(prompt, judge_model)
    elif provider == "openrouter":
        texto_julgamento, telemetria = chamar_openrouter(prompt, judge_model)
    else:
        texto_julgamento, telemetria = chamar_ollama(prompt, judge_model)
    telemetria["wall_duration_seconds"] = round(
        time.perf_counter() - inicio,
        4,
    )

    avaliacao_comparativa = extrair_json(texto_julgamento)
    avaliacao_comparativa.setdefault("id_instancia", payload["id_instancia"])
    avaliacao_comparativa.setdefault("dataset", payload["dataset"])
    avaliacao_comparativa.setdefault("avaliacoes", {})
    avaliacao_comparativa.setdefault("gflow_oracle_3", {})
    return avaliacao_comparativa, payload, telemetria


def normalizar_avaliacoes_por_abordagem(avaliacoes):
    if not isinstance(avaliacoes, dict):
        return {}
    return {
        normalizar_abordagem(abordagem): avaliacao
        for abordagem, avaliacao in avaliacoes.items()
        if normalizar_abordagem(abordagem)
    }


def avaliacao_individual_de_erro(item, exc):
    payload = montar_payload_julgamento(item)
    return {
        "veredito": None,
        "pontuacao": None,
        "resposta_final_modelo": payload["resposta_final_extraida"],
        "resposta_esperada": payload["resposta_esperada_curta"],
        "justificativa_analitica": f"Falha operacional do juiz comparativo: {str(exc)}",
        "tipo_erro": "outro",
        "confianca": 0.0,
    }


def normalizar_float(valor, padrao=0.0):
    try:
        return float(valor)
    except (TypeError, ValueError):
        return padrao


def normalizar_veredito(valor):
    if valor in (0, "0"):
        return 0
    if valor in (1, "1"):
        return 1
    return None


def chave_grupo(grupo):
    return (
        grupo.get("id_instancia"),
        grupo.get("dataset"),
        grupo.get("modelo", ""),
    )


def grupos_auditoria_secundaria(grupos, taxa, seed):
    if taxa <= 0:
        return set()
    quantidade = min(len(grupos), max(1, math.ceil(len(grupos) * taxa)))
    rng = random.Random(seed)
    indices = rng.sample(range(len(grupos)), quantidade)
    return {chave_grupo(grupos[indice]) for indice in indices}


def avaliacao_vazia_grupo(grupo, exc):
    return {
        "id_instancia": grupo.get("id_instancia"),
        "dataset": grupo.get("dataset"),
        "avaliacoes": {},
        "gflow_oracle_3": {},
        "melhor_abordagem": "",
        "ranking_abordagens": [],
        "observacao_comparativa": f"Falha operacional do juiz comparativo: {str(exc)}",
    }


def salvar_resumo_csv(avaliacoes, caminho_csv):
    agregados = {}
    for registro in avaliacoes:
        chave = (registro.get("dataset", ""), registro.get("abordagem", ""))
        bucket = agregados.setdefault(
            chave,
            {
                "total": 0,
                "avaliadas": 0,
                "erros_avaliacao": 0,
                "corretas": 0,
                "pontuacao_soma": 0.0,
                "pontuacao_n": 0,
                "duracao_soma": 0.0,
                "duracao_n": 0,
            },
        )
        avaliacao = registro.get("avaliacao_llm_judge", {})
        bucket["total"] += 1
        veredito = normalizar_veredito(avaliacao.get("veredito"))
        if registro.get("status_avaliacao") != "ok" or veredito is None:
            bucket["erros_avaliacao"] += 1
        else:
            bucket["avaliadas"] += 1
            bucket["corretas"] += veredito
            if avaliacao.get("pontuacao") is not None:
                bucket["pontuacao_soma"] += normalizar_float(avaliacao.get("pontuacao"))
                bucket["pontuacao_n"] += 1
        if registro.get("duracao_segundos") is not None:
            bucket["duracao_soma"] += normalizar_float(registro.get("duracao_segundos"))
            bucket["duracao_n"] += 1

    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo:
        campos = [
            "dataset",
            "abordagem",
            "total",
            "avaliadas",
            "erros_avaliacao",
            "corretas",
            "acuracia",
            "pontuacao_media",
            "duracao_media_segundos",
        ]
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        for (dataset, abordagem), bucket in sorted(agregados.items()):
            avaliadas = bucket["avaliadas"]
            writer.writerow(
                {
                    "dataset": dataset,
                    "abordagem": abordagem,
                    "total": bucket["total"],
                    "avaliadas": avaliadas,
                    "erros_avaliacao": bucket["erros_avaliacao"],
                    "corretas": bucket["corretas"],
                    "acuracia": (
                        round(bucket["corretas"] / avaliadas, 4) if avaliadas else ""
                    ),
                    "pontuacao_media": (
                        round(bucket["pontuacao_soma"] / bucket["pontuacao_n"], 4)
                        if bucket["pontuacao_n"]
                        else ""
                    ),
                    "duracao_media_segundos": (
                        round(bucket["duracao_soma"] / bucket["duracao_n"], 4)
                        if bucket["duracao_n"]
                        else ""
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Avalia resultados experimentais usando LLM-as-a-Judge."
    )
    parser.add_argument("resultados", nargs="+", help="Arquivos JSON ou diretorios de resultados.")
    parser.add_argument(
        "--provider",
        choices=("ollama", "gemini", "openrouter"),
        default=texto("JUDGE_PROVIDER", "ollama"),
    )
    parser.add_argument(
        "--judge-model",
        default=texto("JUDGE_MODEL_NAME"),
        help="Modelo avaliador, preferencialmente mais forte que o SLM alvo.",
    )
    parser.add_argument(
        "--permitir-mesmo-modelo",
        action="store_true",
        help="Permite usar no Ollama o mesmo modelo avaliado, apenas para validacao do fluxo.",
    )
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--limite", type=int, default=None, help="Limite opcional de instancias.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa entre chamadas ao juiz.")
    parser.add_argument(
        "--sem-embaralhar",
        action="store_true",
        help="Mantem a ordem original das abordagens no payload comparativo.",
    )
    parser.add_argument(
        "--sem-auditoria-posicional",
        action="store_true",
        help="Desativa a segunda avaliacao com a ordem das abordagens invertida.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260612,
        help="Seed usada para ordem de apresentacao e amostra do segundo juiz.",
    )
    parser.add_argument(
        "--secondary-provider",
        choices=("ollama", "gemini", "openrouter"),
        default=texto("SECONDARY_JUDGE_PROVIDER", "ollama"),
    )
    parser.add_argument(
        "--secondary-judge-model",
        default=texto("SECONDARY_JUDGE_MODEL"),
        help="Segundo juiz externo usado em uma amostra para medir concordancia.",
    )
    parser.add_argument(
        "--secondary-sample-rate",
        type=float,
        default=0.1,
        help="Fracao das instancias auditadas por um segundo juiz.",
    )
    args = parser.parse_args()

    if not args.judge_model:
        parser.error("Informe --judge-model ou defina JUDGE_MODEL_NAME no ambiente.")
    if not 0 <= args.secondary_sample_rate <= 1:
        parser.error("--secondary-sample-rate deve estar entre 0 e 1.")

    resultados = carregar_resultados(args.resultados)
    grupos = agrupar_por_instancia(resultados)
    modelos_alvo = {grupo.get("modelo") for grupo in grupos if grupo.get("modelo")}
    if (
        args.provider == "ollama"
        and args.judge_model in modelos_alvo
        and not args.permitir_mesmo_modelo
    ):
        parser.error(
            "O juiz deve ser separado do modelo avaliado. Use outro --judge-model ou, "
            "somente para validar o fluxo, passe --permitir-mesmo-modelo."
        )
    if (
        args.secondary_judge_model
        and args.secondary_provider == "ollama"
        and args.secondary_judge_model in modelos_alvo
        and not args.permitir_mesmo_modelo
    ):
        parser.error(
            "O segundo juiz tambem deve ser separado do modelo avaliado."
        )
    if args.limite is not None:
        grupos = grupos[: args.limite]
    grupos_secundarios = grupos_auditoria_secundaria(
        grupos,
        args.secondary_sample_rate if args.secondary_judge_model else 0.0,
        args.seed,
    )

    run_dir = preparar_diretorio_saida(Path(args.output_root))
    jsonl_path = run_dir / "avaliacao_llm_judge.parcial.jsonl"
    json_path = run_dir / "avaliacao_llm_judge.json"
    comparativo_path = run_dir / "avaliacao_llm_judge.comparativa.json"
    csv_path = run_dir / "resumo_avaliacao_llm_judge.csv"

    avaliacoes = []
    avaliacoes_comparativas = []
    with jsonl_path.open("w", encoding="utf-8") as parcial:
        for indice, grupo in enumerate(grupos, start=1):
            seed_grupo = args.seed + hash_estavel(chave_grupo(grupo))
            rng_grupo = random.Random(seed_grupo)
            try:
                avaliacao_comparativa, payload, telemetria_primaria = julgar_grupo(
                    grupo,
                    args.provider,
                    args.judge_model,
                    embaralhar=not args.sem_embaralhar,
                    rng=rng_grupo,
                )
            except Exception as exc:
                telemetria_primaria = {}
                payload = montar_payload_julgamento_comparativo(
                    grupo,
                    embaralhar=not args.sem_embaralhar,
                    rng=random.Random(seed_grupo),
                )
                avaliacao_comparativa = avaliacao_vazia_grupo(grupo, exc)

            ordem_reversa = list(reversed(payload.get("ordem_apresentada", [])))
            avaliacao_reversa = {}
            payload_reverso = {}
            telemetria_reversa = {}
            if not args.sem_auditoria_posicional:
                try:
                    avaliacao_reversa, payload_reverso, telemetria_reversa = julgar_grupo(
                        grupo,
                        args.provider,
                        args.judge_model,
                        embaralhar=False,
                        ordem_abordagens=ordem_reversa,
                    )
                except Exception as exc:
                    payload_reverso = montar_payload_julgamento_comparativo(
                        grupo,
                        embaralhar=False,
                        ordem_abordagens=ordem_reversa,
                    )
                    avaliacao_reversa = avaliacao_vazia_grupo(grupo, exc)

            avaliacao_secundaria = {}
            telemetria_secundaria = {}
            if chave_grupo(grupo) in grupos_secundarios:
                try:
                    avaliacao_secundaria, _, telemetria_secundaria = julgar_grupo(
                        grupo,
                        args.secondary_provider,
                        args.secondary_judge_model,
                        embaralhar=False,
                        ordem_abordagens=payload.get("ordem_apresentada", []),
                    )
                except Exception as exc:
                    avaliacao_secundaria = avaliacao_vazia_grupo(grupo, exc)

            avaliacoes_comparativas.append(
                {
                    "id_instancia": grupo.get("id_instancia"),
                    "modelo": grupo.get("modelo"),
                    "dataset": grupo.get("dataset"),
                    "ordem_apresentada": payload.get("ordem_apresentada", []),
                    "ordem_apresentada_reversa": payload_reverso.get(
                        "ordem_apresentada",
                        [],
                    ),
                    "judge_provider": args.provider,
                    "judge_model": args.judge_model,
                    "secondary_judge_provider": (
                        args.secondary_provider if avaliacao_secundaria else ""
                    ),
                    "secondary_judge_model": (
                        args.secondary_judge_model if avaliacao_secundaria else ""
                    ),
                    "melhor_abordagem": normalizar_abordagem(
                        avaliacao_comparativa.get("melhor_abordagem", "")
                    ),
                    "ranking_abordagens": [
                        normalizar_abordagem(abordagem)
                        for abordagem in avaliacao_comparativa.get("ranking_abordagens", [])
                    ],
                    "observacao_comparativa": avaliacao_comparativa.get("observacao_comparativa", ""),
                    "avaliacao_comparativa": avaliacao_comparativa,
                    "avaliacao_posicao_reversa": avaliacao_reversa,
                    "avaliacao_secundaria": avaliacao_secundaria,
                    "telemetria_juiz_primario": telemetria_primaria,
                    "telemetria_juiz_posicao_reversa": telemetria_reversa,
                    "telemetria_juiz_secundario": telemetria_secundaria,
                }
            )

            avaliacoes_por_abordagem = normalizar_avaliacoes_por_abordagem(
                avaliacao_comparativa.get("avaliacoes", {}) or {}
            )
            avaliacoes_reversas = normalizar_avaliacoes_por_abordagem(
                avaliacao_reversa.get("avaliacoes", {}) or {}
            )
            avaliacoes_secundarias = normalizar_avaliacoes_por_abordagem(
                avaliacao_secundaria.get("avaliacoes", {}) or {}
            )
            for abordagem, item in sorted(grupo["itens"].items()):
                avaliacao = avaliacoes_por_abordagem.get(abordagem)
                status_avaliacao = "ok"
                if not avaliacao:
                    avaliacao = avaliacao_individual_de_erro(
                        item,
                        f"Juiz nao retornou avaliacao para abordagem '{abordagem}'.",
                    )
                    status_avaliacao = "erro"
                elif normalizar_veredito(avaliacao.get("veredito")) is None:
                    status_avaliacao = "invalida"

                registro = {
                    "id_instancia": item.get("id_instancia"),
                    "modelo": item.get("modelo"),
                    "dataset": item.get("dataset"),
                    "abordagem": normalizar_abordagem(item.get("abordagem")),
                    "arquivo_origem": item.get("_arquivo_origem"),
                    "status": item.get("status"),
                    "status_avaliacao": status_avaliacao,
                    "duracao_segundos": item.get("duracao_segundos"),
                    "resposta_final_extraida": extrair_resposta_final(item.get("resposta_gerada", "")),
                    "melhor_abordagem": normalizar_abordagem(
                        avaliacao_comparativa.get("melhor_abordagem", "")
                    ),
                    "ranking_abordagens": [
                        normalizar_abordagem(nome)
                        for nome in avaliacao_comparativa.get("ranking_abordagens", [])
                    ],
                    "observacao_comparativa": avaliacao_comparativa.get("observacao_comparativa", ""),
                    "avaliacao_llm_judge": avaliacao,
                    "avaliacao_llm_judge_posicao_reversa": avaliacoes_reversas.get(
                        abordagem
                    ),
                    "avaliacao_llm_judge_secundario": avaliacoes_secundarias.get(
                        abordagem
                    ),
                    "gflow_oracle_3": (
                        avaliacao_comparativa.get("gflow_oracle_3", {})
                        if abordagem == "gflow"
                        else {}
                    ),
                    "judge_provider": args.provider,
                    "judge_model": args.judge_model,
                    "secondary_judge_provider": (
                        args.secondary_provider if avaliacao_secundaria else ""
                    ),
                    "secondary_judge_model": (
                        args.secondary_judge_model if avaliacao_secundaria else ""
                    ),
                    "telemetria_juiz_primario": telemetria_primaria,
                    "telemetria_juiz_posicao_reversa": telemetria_reversa,
                    "telemetria_juiz_secundario": telemetria_secundaria,
                }
                avaliacoes.append(registro)
                parcial.write(json.dumps(registro, ensure_ascii=False) + "\n")

            parcial.flush()

            print(
                f"[{indice}/{len(grupos)}] {grupo.get('dataset')} | "
                f"{grupo.get('id_instancia')} | abordagens={','.join(sorted(grupo['itens'].keys()))}"
            )
            if args.sleep:
                time.sleep(args.sleep)

    with json_path.open("w", encoding="utf-8") as arquivo:
        json.dump(avaliacoes, arquivo, ensure_ascii=False, indent=2)

    with comparativo_path.open("w", encoding="utf-8") as arquivo:
        json.dump(avaliacoes_comparativas, arquivo, ensure_ascii=False, indent=2)

    salvar_resumo_csv(avaliacoes, csv_path)
    print(f"Avaliacao salva em: {json_path}")
    print(f"Avaliacao comparativa salva em: {comparativo_path}")
    print(f"Resumo salvo em: {csv_path}")


if __name__ == "__main__":
    main()
