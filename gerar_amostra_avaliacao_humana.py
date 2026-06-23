import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from metricas_avaliacao import referencia_curta
from processar_resultados import carregar_resultados, normalizar_abordagem
from util_experimento import extrair_resposta_final


def amostrar_instancias(resultados, quantidade, seed):
    grupos = defaultdict(list)
    for item in resultados:
        chave = (
            item.get("id_instancia"),
            item.get("dataset"),
            item.get("modelo", ""),
        )
        grupos[chave].append(item)

    por_dataset = defaultdict(list)
    for chave in grupos:
        por_dataset[chave[1]].append(chave)

    rng = random.Random(seed)
    selecionadas = []
    datasets = sorted(por_dataset)
    base = quantidade // max(len(datasets), 1)
    resto = quantidade % max(len(datasets), 1)
    for indice, dataset in enumerate(datasets):
        chaves = list(por_dataset[dataset])
        rng.shuffle(chaves)
        alvo = min(len(chaves), base + (1 if indice < resto else 0))
        selecionadas.extend(chaves[:alvo])

    if len(selecionadas) < quantidade:
        restantes = [chave for chave in grupos if chave not in set(selecionadas)]
        rng.shuffle(restantes)
        selecionadas.extend(restantes[: quantidade - len(selecionadas)])
    rng.shuffle(selecionadas)
    return [grupos[chave] for chave in selecionadas]


def referencia_item(item):
    return referencia_curta(item)


def main():
    parser = argparse.ArgumentParser(
        description="Gera uma amostra cega para avaliacao humana."
    )
    parser.add_argument("resultados", nargs="+")
    parser.add_argument("--instancias", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--output-dir", default="avaliacao_humana")
    args = parser.parse_args()

    resultados = carregar_resultados(args.resultados)
    grupos = amostrar_instancias(resultados, args.instancias, args.seed)
    if not grupos:
        raise RuntimeError(
            "Nenhum resultado valido foi encontrado para gerar a amostra humana."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    linhas = []
    chave = []
    rng = random.Random(args.seed)
    contador = 1
    for grupo in grupos:
        respostas = list(grupo)
        rng.shuffle(respostas)
        for item in respostas:
            codigo = f"H{contador:05d}"
            contador += 1
            linhas.append(
                {
                    "codigo_resposta": codigo,
                    "dataset": item.get("dataset"),
                    "pergunta": item.get("pergunta"),
                    "referencia": referencia_item(item),
                    "respostas_corretas_truthfulqa": json.dumps(
                        item.get("respostas_corretas", []),
                        ensure_ascii=False,
                    ),
                    "respostas_incorretas_truthfulqa": json.dumps(
                        item.get("respostas_incorretas", []),
                        ensure_ascii=False,
                    ),
                    "resposta_modelo": extrair_resposta_final(
                        item.get("resposta_gerada", "")
                    ),
                    "human_veredito": "",
                    "human_truthfulness": "",
                    "human_informativeness": "",
                    "observacao_humana": "",
                }
            )
            chave.append(
                {
                    "codigo_resposta": codigo,
                    "id_instancia": item.get("id_instancia"),
                    "dataset": item.get("dataset"),
                    "modelo": item.get("modelo", ""),
                    "abordagem": normalizar_abordagem(item.get("abordagem")),
                }
            )

    csv_path = output_dir / "amostra_avaliacao_humana.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)
    (output_dir / "chave_avaliacao_humana.json").write_text(
        json.dumps(chave, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "instrucoes.txt").write_text(
        "Preencha 0 ou 1 em human_veredito. Para TruthfulQA, preencha tambem "
        "human_truthfulness e human_informativeness. Nao consulte a chave JSON "
        "antes de concluir a avaliacao.\n",
        encoding="utf-8",
    )
    print(f"Amostra humana: {csv_path}")


if __name__ == "__main__":
    main()
