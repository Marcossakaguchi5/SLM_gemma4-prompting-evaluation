"""Avalia respostas objetivas já geradas, sem chamar um modelo externo."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from metricas_avaliacao import answer_match_objetivo
from processar_resultados import carregar_resultados, normalizar_abordagem


OUTPUT_ROOT = Path("avaliacoes_deterministicas")


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


def classificacao_recomendada(item, metrica):
    dataset = str(item.get("dataset", "")).lower()
    veredito = metrica.get("answer_match_objetivo")
    if "truthful" in dataset:
        return "llm_juiz_principal", True
    if any(nome in dataset for nome in ("gsm8k", "arc", "math")):
        return (
            ("deterministica", False)
            if veredito is not None
            else ("llm_juiz_fallback", True)
        )
    return "llm_juiz_principal", True


def avaliar_item(item):
    metrica = answer_match_objetivo(item, item.get("resposta_gerada", ""))
    if "truthful" in str(item.get("dataset", "")).lower():
        # TruthfulQA exige equivalência semântica; matching textual não é veredito.
        metrica["answer_match_objetivo"] = None
        metrica["metrica_objetiva"] = ""
    fonte, necessita_juiz = classificacao_recomendada(item, metrica)
    return {
        "id_instancia": item.get("id_instancia"),
        "modelo": item.get("modelo", ""),
        "dataset": item.get("dataset"),
        "abordagem": normalizar_abordagem(item.get("abordagem")),
        "status": item.get("status"),
        "resposta_final_extraida": metrica.get("resposta_final"),
        "referencia_curta": metrica.get("referencia_curta"),
        "veredito_deterministico": metrica.get("answer_match_objetivo"),
        "exact_match": metrica.get("exact_match"),
        "numeric_match": metrica.get("numeric_match"),
        "symbolic_equivalence": metrica.get("symbolic_equivalence"),
        "metrica_deterministica": metrica.get("metrica_objetiva"),
        "fonte_avaliacao_recomendada": fonte,
        "necessita_llm_juiz": necessita_juiz,
        "arquivo_origem": item.get("_arquivo_origem"),
    }


def salvar_csv(path, linhas):
    if not linhas:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


def gerar_resumo(linhas):
    grupos = defaultdict(list)
    for linha in linhas:
        grupos[(linha["dataset"], linha["abordagem"])].append(linha)
    resumo = []
    for (dataset, abordagem), grupo in sorted(grupos.items()):
        determinados = [
            linha["veredito_deterministico"]
            for linha in grupo
            if linha["veredito_deterministico"] is not None
        ]
        resumo.append(
            {
                "dataset": dataset,
                "abordagem": abordagem,
                "total": len(grupo),
                "deterministicamente_avaliadas": len(determinados),
                "corretas_deterministicas": sum(determinados),
                "pendentes_llm_juiz": sum(
                    linha["necessita_llm_juiz"] for linha in grupo
                ),
            }
        )
    return resumo


def main():
    parser = argparse.ArgumentParser(
        description="Avalia GSM8K, ARC e MATH por regras determinísticas."
    )
    parser.add_argument("resultados", nargs="+", help="JSONs ou diretórios brutos.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    resultados = carregar_resultados(args.resultados)
    linhas = [avaliar_item(item) for item in resultados]
    run_dir = preparar_diretorio_saida(Path(args.output_root))
    with (run_dir / "avaliacao_deterministica.json").open("w", encoding="utf-8") as arquivo:
        json.dump(linhas, arquivo, ensure_ascii=False, indent=2)
    salvar_csv(run_dir / "avaliacao_deterministica.csv", linhas)
    salvar_csv(run_dir / "resumo_avaliacao_deterministica.csv", gerar_resumo(linhas))
    print(f"Avaliacao deterministica salva em: {run_dir}")


if __name__ == "__main__":
    main()
