import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

from metricas_avaliacao import (
    answer_match_objetivo,
    cohens_kappa,
    correcao_holm,
    intervalo_bootstrap_binario,
    mcnemar_exato,
)
from util_experimento import extrair_resposta_final


OUTPUT_ROOT = Path("analises_resultados")


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


def descobrir_jsons(caminhos):
    arquivos = []
    for caminho_bruto in caminhos:
        caminho = Path(caminho_bruto)
        if caminho.is_dir():
            arquivos.extend(sorted(caminho.rglob("*.json")))
        else:
            arquivos.append(caminho)
    return arquivos


def carregar_lista_json(path):
    with path.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    if isinstance(dados, list):
        return dados
    if isinstance(dados, dict) and isinstance(dados.get("resultados"), list):
        return dados["resultados"]
    return []


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
    return chave if chave in {"base", "cot", "for", "gflow"} else abordagem


def carregar_resultados(caminhos):
    resultados = []
    for arquivo in descobrir_jsons(caminhos):
        if any(
            marcador in arquivo.name
            for marcador in ("avaliacao_llm_judge", "metricas", "resumo")
        ):
            continue
        for item in carregar_lista_json(arquivo):
            if "resposta_gerada" not in item or "abordagem" not in item:
                continue
            item = dict(item)
            item["_arquivo_origem"] = str(arquivo)
            resultados.append(item)
    return resultados


def chave_registro(item):
    return (
        item.get("id_instancia"),
        item.get("dataset"),
        item.get("modelo", ""),
        normalizar_abordagem(item.get("abordagem")),
    )


def carregar_avaliacoes(caminhos):
    avaliacoes = {}
    for arquivo in descobrir_jsons(caminhos or []):
        for item in carregar_lista_json(arquivo):
            if not item.get("avaliacao_llm_judge"):
                continue
            avaliacoes[chave_registro(item)] = {
                "primaria": item.get("avaliacao_llm_judge") or {},
                "reversa": item.get("avaliacao_llm_judge_posicao_reversa") or {},
                "secundaria": item.get("avaliacao_llm_judge_secundario") or {},
                "oracle_3": item.get("gflow_oracle_3") or {},
                "status_avaliacao": item.get("status_avaliacao", "ok"),
                "melhor_abordagem": item.get("melhor_abordagem", ""),
                "ranking_abordagens": item.get("ranking_abordagens", []),
                "observacao_comparativa": item.get("observacao_comparativa", ""),
                "judge_model": item.get("judge_model", ""),
                "secondary_judge_model": item.get("secondary_judge_model", ""),
            }
    return avaliacoes


def carregar_avaliacoes_deterministicas(caminhos):
    avaliacoes = {}
    for arquivo in descobrir_jsons(caminhos or []):
        for item in carregar_lista_json(arquivo):
            if "veredito_deterministico" not in item:
                continue
            avaliacoes[chave_registro(item)] = item
    return avaliacoes


def carregar_avaliacao_humana(csv_path, chave_path):
    if not csv_path:
        return {}
    mapeamento = {}
    if chave_path:
        with Path(chave_path).open("r", encoding="utf-8") as arquivo:
            for item in json.load(arquivo):
                mapeamento[item["codigo_resposta"]] = item

    avaliacoes = {}
    with Path(csv_path).open("r", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            codigo = linha.get("codigo_resposta")
            chave_item = mapeamento.get(codigo, linha)
            chave = (
                chave_item.get("id_instancia"),
                chave_item.get("dataset"),
                chave_item.get("modelo", ""),
                normalizar_abordagem(chave_item.get("abordagem")),
            )
            avaliacoes[chave] = {
                "veredito": normalizar_binario(linha.get("human_veredito")),
                "truthfulness": normalizar_binario(linha.get("human_truthfulness")),
                "informativeness": normalizar_binario(
                    linha.get("human_informativeness")
                ),
            }
    return avaliacoes


def normalizar_binario(valor):
    if valor in (0, "0", False):
        return 0
    if valor in (1, "1", True):
        return 1
    return None


def valor_float(valor):
    try:
        if valor in (None, ""):
            return None
        return float(valor)
    except (TypeError, ValueError):
        return None


def extrair_oracle_objetivo(item):
    if normalizar_abordagem(item.get("abordagem")) != "gflow":
        return None
    if "truthful" in str(item.get("dataset", "")).lower():
        return None
    resultados = []
    for resposta in (item.get("rastros_execucao") or {}).values():
        metrica = answer_match_objetivo(item, resposta)
        resultados.append(metrica.get("answer_match_objetivo"))
    validos = [valor for valor in resultados if valor is not None]
    return max(validos) if validos else None


def escolher_answer_match(item, objetiva, avaliacao):
    dataset = str(item.get("dataset", "")).lower()
    objetivo = objetiva.get("answer_match_objetivo")
    juiz = normalizar_binario(avaliacao.get("veredito"))
    if any(nome in dataset for nome in ("gsm8k", "arc", "math")):
        return objetivo if objetivo is not None else juiz
    return juiz if juiz is not None else objetivo


def normalizar_linha(
    item,
    avaliacoes,
    avaliacoes_humanas=None,
    avaliacoes_deterministicas=None,
):
    abordagem = normalizar_abordagem(item.get("abordagem"))
    pacote = avaliacoes.get(chave_registro(item), {})
    primaria = pacote.get("primaria", {})
    reversa = pacote.get("reversa", {})
    secundaria = pacote.get("secundaria", {})
    deterministica = (avaliacoes_deterministicas or {}).get(chave_registro(item))
    objetiva = deterministica or answer_match_objetivo(item, item.get("resposta_gerada", ""))
    if deterministica:
        objetiva = {
            "resposta_final": deterministica.get("resposta_final_extraida"),
            "referencia_curta": deterministica.get("referencia_curta"),
            "exact_match": deterministica.get("exact_match"),
            "numeric_match": deterministica.get("numeric_match"),
            "symbolic_equivalence": deterministica.get("symbolic_equivalence"),
            "answer_match_objetivo": deterministica.get("veredito_deterministico"),
            "metrica_objetiva": deterministica.get("metrica_deterministica", ""),
        }
    answer_match = escolher_answer_match(item, objetiva, primaria)
    oracle_objetivo = extrair_oracle_objetivo(item)
    oracle_juiz = normalizar_binario(
        (pacote.get("oracle_3") or {}).get("veredito")
    )
    oracle_at_3 = oracle_objetivo if oracle_objetivo is not None else oracle_juiz
    telemetria = item.get("telemetria") or {}
    humana = (avaliacoes_humanas or {}).get(chave_registro(item), {})

    return {
        "id_instancia": item.get("id_instancia"),
        "modelo": item.get("modelo", ""),
        "dataset": item.get("dataset"),
        "abordagem": abordagem,
        "status": item.get("status"),
        "status_avaliacao": pacote.get("status_avaliacao", ""),
        "resposta_final_extraida": objetiva.get("resposta_final"),
        "referencia_curta": objetiva.get("referencia_curta"),
        "exact_match": objetiva.get("exact_match"),
        "numeric_match": objetiva.get("numeric_match"),
        "symbolic_equivalence": objetiva.get("symbolic_equivalence"),
        "answer_match_objetivo": objetiva.get("answer_match_objetivo"),
        "answer_match": answer_match,
        "correta": answer_match,
        "metrica_objetiva": objetiva.get("metrica_objetiva"),
        "fonte_avaliacao_principal": (
            deterministica.get("fonte_avaliacao_recomendada")
            if deterministica
            else ""
        ),
        "necessita_llm_juiz": (
            deterministica.get("necessita_llm_juiz") if deterministica else ""
        ),
        "pontuacao_juiz": valor_float(primaria.get("pontuacao")),
        "tipo_erro": primaria.get("tipo_erro", ""),
        "confianca_juiz": valor_float(primaria.get("confianca")),
        "truthfulness": normalizar_binario(primaria.get("truthfulness")),
        "informativeness": normalizar_binario(primaria.get("informativeness")),
        "oracle_at_3": oracle_at_3,
        "veredito_juiz_primario": normalizar_binario(primaria.get("veredito")),
        "veredito_juiz_posicao_reversa": normalizar_binario(
            reversa.get("veredito")
        ),
        "veredito_juiz_secundario": normalizar_binario(
            secundaria.get("veredito")
        ),
        "human_veredito": humana.get("veredito"),
        "human_truthfulness": humana.get("truthfulness"),
        "human_informativeness": humana.get("informativeness"),
        "duracao_segundos": valor_float(item.get("duracao_segundos")),
        "numero_chamadas_slm": int(
            item.get("numero_chamadas_slm", 3 if abordagem == "gflow" else 1)
        ),
        "input_tokens": int(telemetria.get("input_tokens", 0) or 0),
        "output_tokens": int(telemetria.get("output_tokens", 0) or 0),
        "total_tokens": int(telemetria.get("total_tokens", 0) or 0),
        "tokens_per_second": valor_float(telemetria.get("tokens_per_second")),
        "model_total_duration_seconds": valor_float(
            telemetria.get("model_total_duration_seconds")
        ),
        "melhor_abordagem": pacote.get("melhor_abordagem", ""),
        "ranking_abordagens": json.dumps(
            pacote.get("ranking_abordagens", []),
            ensure_ascii=False,
        ),
        "judge_model": pacote.get("judge_model", ""),
        "secondary_judge_model": pacote.get("secondary_judge_model", ""),
        "arquivo_origem": item.get("_arquivo_origem"),
    }


def media_valida(linhas, campo):
    valores = [valor_float(linha.get(campo)) for linha in linhas]
    valores = [valor for valor in valores if valor is not None]
    return round(sum(valores) / len(valores), 4) if valores else ""


def taxa_binaria(linhas, campo):
    valores = [normalizar_binario(linha.get(campo)) for linha in linhas]
    valores = [valor for valor in valores if valor is not None]
    return (round(sum(valores) / len(valores), 4), len(valores)) if valores else ("", 0)


def agregar(linhas, campos_chave, bootstrap_iteracoes, seed):
    buckets = defaultdict(list)
    for linha in linhas:
        chave = tuple(linha.get(campo, "") for campo in campos_chave)
        buckets[chave].append(linha)

    saida = []
    for indice, (chave, grupo) in enumerate(sorted(buckets.items())):
        answer_match, avaliadas = taxa_binaria(grupo, "answer_match")
        exact, exact_n = taxa_binaria(grupo, "exact_match")
        symbolic, symbolic_n = taxa_binaria(grupo, "symbolic_equivalence")
        truthfulness, truth_n = taxa_binaria(grupo, "truthfulness")
        informativeness, info_n = taxa_binaria(grupo, "informativeness")
        oracle, oracle_n = taxa_binaria(grupo, "oracle_at_3")
        valores_bootstrap = [
            normalizar_binario(linha.get("answer_match")) for linha in grupo
        ]
        ci_low, ci_high = intervalo_bootstrap_binario(
            valores_bootstrap,
            iteracoes=bootstrap_iteracoes,
            seed=seed + indice,
        )
        duracao_media = media_valida(grupo, "duracao_segundos")
        chamadas_media = media_valida(grupo, "numero_chamadas_slm")
        registro = dict(zip(campos_chave, chave))
        registro.update(
            {
                "total": len(grupo),
                "status_ok": sum(linha.get("status") == "ok" for linha in grupo),
                "erros_operacionais": sum(
                    linha.get("status") != "ok" for linha in grupo
                ),
                "erros_avaliacao": sum(
                    linha.get("status_avaliacao") in {"erro", "invalida"}
                    for linha in grupo
                ),
                "avaliadas": avaliadas,
                "corretas": sum(
                    normalizar_binario(linha.get("answer_match")) or 0
                    for linha in grupo
                ),
                "acuracia": answer_match,
                "answer_match_rate": answer_match,
                "accuracy_ci95_low": "" if ci_low is None else ci_low,
                "accuracy_ci95_high": "" if ci_high is None else ci_high,
                "exact_match_rate": exact,
                "exact_match_n": exact_n,
                "symbolic_equivalence_rate": symbolic,
                "symbolic_equivalence_n": symbolic_n,
                "truthfulness_rate": truthfulness,
                "truthfulness_n": truth_n,
                "informativeness_rate": informativeness,
                "informativeness_n": info_n,
                "oracle_at_3_rate": oracle,
                "oracle_at_3_n": oracle_n,
                "pontuacao_media_juiz": media_valida(grupo, "pontuacao_juiz"),
                "duracao_media_segundos": duracao_media,
                "duracao_total_segundos": round(
                    sum(valor_float(linha.get("duracao_segundos")) or 0 for linha in grupo),
                    4,
                ),
                "chamadas_slm_media": chamadas_media,
                "chamadas_slm_total": sum(
                    int(linha.get("numero_chamadas_slm") or 0) for linha in grupo
                ),
                "input_tokens_media": media_valida(grupo, "input_tokens"),
                "output_tokens_media": media_valida(grupo, "output_tokens"),
                "tokens_per_second_media": media_valida(
                    grupo,
                    "tokens_per_second",
                ),
                "accuracy_per_second": (
                    round(float(answer_match) / float(duracao_media), 6)
                    if answer_match != "" and duracao_media not in ("", 0)
                    else ""
                ),
            }
        )
        saida.append(registro)
    return saida


def adicionar_comparacao_base(metricas, campos_contexto):
    bases = {}
    for linha in metricas:
        if linha.get("abordagem") == "base":
            chave = tuple(linha.get(campo, "") for campo in campos_contexto)
            bases[chave] = linha

    for linha in metricas:
        chave = tuple(linha.get(campo, "") for campo in campos_contexto)
        base = bases.get(chave)
        acuracia = valor_float(linha.get("acuracia"))
        acuracia_base = valor_float(base.get("acuracia")) if base else None
        chamadas = valor_float(linha.get("chamadas_slm_media"))
        chamadas_base = valor_float(base.get("chamadas_slm_media")) if base else None
        if acuracia is None or acuracia_base is None:
            linha["delta_acuracia_vs_base"] = ""
            linha["gain_per_extra_call"] = ""
            continue
        delta = acuracia - acuracia_base
        linha["delta_acuracia_vs_base"] = round(delta, 4)
        chamadas_extras = (
            chamadas - chamadas_base
            if chamadas is not None and chamadas_base is not None
            else 0
        )
        linha["gain_per_extra_call"] = (
            round(delta / chamadas_extras, 6) if chamadas_extras > 0 else ""
        )


def adicionar_rotulos_condicao(metricas):
    """Cria um rotulo inequivoco quando ha mais de um modelo na rodada."""
    for linha in metricas:
        modelo = str(linha.get("modelo", "")).strip()
        abordagem = str(linha.get("abordagem", "")).strip()
        linha["condicao"] = f"{modelo} / {abordagem}" if modelo else abordagem


def intervalo_bootstrap_delta(pares, iteracoes, seed):
    if not pares:
        return None, None
    rng = random.Random(seed)
    n = len(pares)
    deltas = []
    for _ in range(iteracoes):
        amostra = [pares[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(estrategia - base for base, estrategia in amostra) / n)
    deltas.sort()
    return (
        round(deltas[int(0.025 * iteracoes)], 4),
        round(deltas[min(iteracoes - 1, int(0.975 * iteracoes))], 4),
    )


def comparacao_vs_base(linhas, bootstrap_iteracoes, seed):
    por_instancia = defaultdict(dict)
    for linha in linhas:
        resposta = normalizar_binario(linha.get("answer_match"))
        if resposta is None:
            continue
        chave = (
            linha.get("modelo", ""),
            linha.get("dataset"),
            linha.get("id_instancia"),
        )
        por_instancia[chave][linha.get("abordagem")] = resposta

    grupos = defaultdict(list)
    for (modelo, dataset, _), respostas in por_instancia.items():
        if "base" not in respostas:
            continue
        for abordagem, resposta in respostas.items():
            if abordagem != "base":
                grupos[(modelo, dataset, abordagem)].append(
                    (respostas["base"], resposta)
                )

    saida = []
    for indice, ((modelo, dataset, abordagem), pares) in enumerate(sorted(grupos.items())):
        wins = sum(base == 0 and estrategia == 1 for base, estrategia in pares)
        losses = sum(base == 1 and estrategia == 0 for base, estrategia in pares)
        ties = len(pares) - wins - losses
        ci_low, ci_high = intervalo_bootstrap_delta(
            pares,
            bootstrap_iteracoes,
            seed + indice,
        )
        saida.append(
            {
                "modelo": modelo,
                "dataset": dataset,
                "abordagem": abordagem,
                "n_pareado": len(pares),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": round(wins / len(pares), 4),
                "tie_rate": round(ties / len(pares), 4),
                "loss_rate": round(losses / len(pares), 4),
                "delta_pareado": round(
                    sum(estrategia - base for base, estrategia in pares) / len(pares),
                    4,
                ),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "mcnemar_p_exato": mcnemar_exato(wins, losses),
            }
        )
    return correcao_holm(saida)


def comparacao_entre_modelos_base(linhas, bootstrap_iteracoes, seed):
    """Compara, de forma pareada, apenas as condicoes base de cada modelo."""
    por_instancia = defaultdict(dict)
    for linha in linhas:
        if linha.get("abordagem") != "base":
            continue
        resposta = normalizar_binario(linha.get("answer_match"))
        modelo = linha.get("modelo", "")
        if resposta is None or not modelo:
            continue
        chave = (linha.get("dataset"), linha.get("id_instancia"))
        por_instancia[chave][modelo] = resposta

    grupos = defaultdict(list)
    for (dataset, _), respostas in por_instancia.items():
        for modelo_referencia, modelo_comparado in combinations(sorted(respostas), 2):
            grupos[(dataset, modelo_referencia, modelo_comparado)].append(
                (respostas[modelo_referencia], respostas[modelo_comparado])
            )

    saida = []
    for indice, ((dataset, referencia, comparado), pares) in enumerate(
        sorted(grupos.items())
    ):
        wins = sum(base == 0 and estrategia == 1 for base, estrategia in pares)
        losses = sum(base == 1 and estrategia == 0 for base, estrategia in pares)
        ties = len(pares) - wins - losses
        ci_low, ci_high = intervalo_bootstrap_delta(
            pares,
            bootstrap_iteracoes,
            seed + indice,
        )
        saida.append(
            {
                "dataset": dataset,
                "modelo_referencia": referencia,
                "modelo_comparado": comparado,
                "n_pareado": len(pares),
                "wins_modelo_comparado": wins,
                "ties": ties,
                "losses_modelo_comparado": losses,
                "win_rate_modelo_comparado": round(wins / len(pares), 4),
                "tie_rate": round(ties / len(pares), 4),
                "loss_rate_modelo_comparado": round(losses / len(pares), 4),
                "delta_pareado_comparado_menos_referencia": round(
                    sum(comparado_valor - referencia_valor for referencia_valor, comparado_valor in pares)
                    / len(pares),
                    4,
                ),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "mcnemar_p_exato": mcnemar_exato(wins, losses),
            }
        )
    return correcao_holm(saida)


def acordo_percentual(valores_a, valores_b):
    pares = [
        (a, b)
        for a, b in zip(valores_a, valores_b)
        if a is not None and b is not None
    ]
    if not pares:
        return None, 0
    return round(sum(a == b for a, b in pares) / len(pares), 4), len(pares)


def calcular_concordancias(linhas):
    especificacoes = [
        (
            "primary_vs_reversed_position",
            "veredito_juiz_primario",
            "veredito_juiz_posicao_reversa",
        ),
        (
            "primary_vs_secondary_judge",
            "veredito_juiz_primario",
            "veredito_juiz_secundario",
        ),
        ("judge_vs_human", "veredito_juiz_primario", "human_veredito"),
        ("truthfulness_judge_vs_human", "truthfulness", "human_truthfulness"),
        (
            "informativeness_judge_vs_human",
            "informativeness",
            "human_informativeness",
        ),
    ]
    saida = []
    for nome, campo_a, campo_b in especificacoes:
        valores_a = [normalizar_binario(linha.get(campo_a)) for linha in linhas]
        valores_b = [normalizar_binario(linha.get(campo_b)) for linha in linhas]
        acordo, n = acordo_percentual(valores_a, valores_b)
        if not n:
            continue
        kappa = cohens_kappa(valores_a, valores_b)
        registro = {
            "comparacao": nome,
            "n": n,
            "agreement_rate": acordo,
            "cohens_kappa": kappa,
        }
        if nome == "primary_vs_reversed_position":
            registro["position_bias_rate"] = round(1 - acordo, 4)
        saida.append(registro)
    return saida


def auditoria_posicional_por_abordagem(linhas):
    grupos = defaultdict(list)
    for linha in linhas:
        primario = normalizar_binario(linha.get("veredito_juiz_primario"))
        reverso = normalizar_binario(
            linha.get("veredito_juiz_posicao_reversa")
        )
        if primario is None or reverso is None:
            continue
        grupos[
            (
                linha.get("modelo", ""),
                linha.get("dataset"),
                linha.get("abordagem"),
            )
        ].append((primario, reverso))

    saida = []
    for (modelo, dataset, abordagem), pares in sorted(grupos.items()):
        divergencias = sum(a != b for a, b in pares)
        saida.append(
            {
                "modelo": modelo,
                "dataset": dataset,
                "abordagem": abordagem,
                "n": len(pares),
                "position_bias_count": divergencias,
                "position_bias_rate": round(divergencias / len(pares), 4),
                "agreement_rate": round(1 - divergencias / len(pares), 4),
                "cohens_kappa": cohens_kappa(
                    [a for a, _ in pares],
                    [b for _, b in pares],
                ),
            }
        )
    return saida


def salvar_csv(path, linhas):
    if not linhas:
        path.write_text("", encoding="utf-8")
        return
    campos = []
    for linha in linhas:
        for campo in linha:
            if campo not in campos:
                campos.append(campo)
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(linhas)


def main():
    parser = argparse.ArgumentParser(
        description="Processa resultados, avaliacoes objetivas e auditorias dos juizes."
    )
    parser.add_argument(
        "resultados",
        nargs="+",
        help="Arquivos JSON ou diretorios de resultados brutos.",
    )
    parser.add_argument(
        "--avaliacoes",
        nargs="*",
        default=[],
        help="Arquivos/diretorios de avaliacoes LLM-as-a-Judge.",
    )
    parser.add_argument(
        "--avaliacoes-deterministicas",
        nargs="*",
        default=[],
        help="Arquivos/diretorios produzidos por avaliar_deterministico.py.",
    )
    parser.add_argument("--avaliacao-humana", help="CSV preenchido por avaliadores humanos.")
    parser.add_argument("--chave-humana", help="JSON de mapeamento da amostra humana.")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    resultados = carregar_resultados(args.resultados)
    avaliacoes = carregar_avaliacoes(args.avaliacoes)
    avaliacoes_deterministicas = carregar_avaliacoes_deterministicas(
        args.avaliacoes_deterministicas
    )
    humanas = carregar_avaliacao_humana(
        args.avaliacao_humana,
        args.chave_humana,
    )
    linhas = [
        normalizar_linha(item, avaliacoes, humanas, avaliacoes_deterministicas)
        for item in resultados
    ]

    run_dir = preparar_diretorio_saida(Path(args.output_root))
    salvar_csv(run_dir / "respostas_normalizadas.csv", linhas)

    metricas_por_abordagem = agregar(
        linhas,
        ["modelo", "abordagem"],
        args.bootstrap_iterations,
        args.seed,
    )
    metricas_por_dataset_abordagem = agregar(
        linhas,
        ["modelo", "dataset", "abordagem"],
        args.bootstrap_iterations,
        args.seed,
    )
    metricas_por_modelo_dataset_abordagem = agregar(
        linhas,
        ["modelo", "dataset", "abordagem"],
        args.bootstrap_iterations,
        args.seed,
    )
    adicionar_comparacao_base(
        metricas_por_dataset_abordagem,
        ["modelo", "dataset"],
    )
    adicionar_comparacao_base(
        metricas_por_modelo_dataset_abordagem,
        ["modelo", "dataset"],
    )
    adicionar_rotulos_condicao(metricas_por_abordagem)
    adicionar_rotulos_condicao(metricas_por_dataset_abordagem)
    adicionar_rotulos_condicao(metricas_por_modelo_dataset_abordagem)

    win_tie_loss = comparacao_vs_base(
        linhas,
        args.bootstrap_iterations,
        args.seed,
    )
    comparacao_modelos = comparacao_entre_modelos_base(
        linhas,
        args.bootstrap_iterations,
        args.seed,
    )
    concordancias = calcular_concordancias(linhas)
    position_bias = auditoria_posicional_por_abordagem(linhas)
    truthfulqa = [
        linha
        for linha in metricas_por_modelo_dataset_abordagem
        if "truthful" in str(linha.get("dataset", "")).lower()
    ]
    oracle = [
        linha
        for linha in metricas_por_modelo_dataset_abordagem
        if linha.get("abordagem") == "gflow"
    ]

    arquivos = {
        "metricas_por_abordagem.csv": metricas_por_abordagem,
        "metricas_por_dataset_abordagem.csv": metricas_por_dataset_abordagem,
        "metricas_por_modelo_dataset_abordagem.csv": metricas_por_modelo_dataset_abordagem,
        "win_tie_loss_vs_baseline.csv": win_tie_loss,
        "comparacao_pareada_modelos_base.csv": comparacao_modelos,
        "auditoria_position_bias.csv": position_bias,
        "concordancia_avaliadores.csv": concordancias,
        "metricas_truthfulqa.csv": truthfulqa,
        "metricas_gflow_oracle.csv": oracle,
    }
    for nome, dados in arquivos.items():
        salvar_csv(run_dir / nome, dados)

    resumo = {
        "total_respostas": len(linhas),
        "total_avaliacoes_llm_judge": len(avaliacoes),
        "total_avaliacoes_deterministicas": len(avaliacoes_deterministicas),
        "total_avaliacoes_humanas": len(humanas),
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "arquivos_gerados": ["respostas_normalizadas.csv", *arquivos.keys()],
    }
    (run_dir / "resumo_geral.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Analise salva em: {run_dir}")


if __name__ == "__main__":
    main()
