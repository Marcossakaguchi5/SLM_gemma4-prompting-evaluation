"""Pipeline 1: gera os dados brutos dos experimentos configurados no .env."""

import argparse
from datetime import datetime

from configuracao.ambiente import inteiro, lista, texto
from configuracao.prompts import PROMPT_VERSION
from .util import (
    EXPERIMENTOS,
    EXPERIMENTOS_PRINCIPAIS,
    ambiente_com_configuracao,
    atualizar_manifesto,
    criar_execucao,
    executar_comando,
    python_do_projeto,
    salvar_execucao_atual,
)


def obter_experimentos():
    selecionados = tuple(
        dict.fromkeys(lista("PIPELINE_EXPERIMENTS", EXPERIMENTOS_PRINCIPAIS))
    )
    desconhecidos = set(selecionados) - set(EXPERIMENTOS)
    if desconhecidos:
        raise ValueError(
            "PIPELINE_EXPERIMENTS contem experimentos invalidos: "
            f"{', '.join(sorted(desconhecidos))}."
        )
    return selecionados


def main():
    parser = argparse.ArgumentParser(
        description="Gera os dados dos experimentos definidos no .env."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    amostras = inteiro("EXPERIMENT_NUM_SAMPLES", 100, minimo=1)
    seed = inteiro("EXPERIMENT_SEED", 20260612)
    modelo = texto("SLM_MODEL_NAME", "gemma4:e4b")
    tarefas = inteiro("EXPERIMENT_TASK_CONCURRENCY", 16, minimo=1)
    chamadas = inteiro("EXPERIMENT_CALL_CONCURRENCY", 4, minimo=1)
    experimentos = obter_experimentos()
    diretorio = criar_execucao()

    if args.dry_run:
        print(f"Execucao planejada: {diretorio}")
    else:
        diretorio.mkdir(parents=True)

    manifesto = {
        "tipo": "pipeline_geracao",
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "status": "em_andamento",
        "configuracao": {
            "modelo": modelo,
            "seed": seed,
            "amostras": amostras,
            "task_concurrency": tarefas,
            "call_concurrency": chamadas,
            "prompt_version": PROMPT_VERSION,
        },
        "experimentos": {},
    }
    caminho_manifesto = diretorio / "manifesto_pipeline.json"
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)

    ambiente_base = ambiente_com_configuracao(
        SLM_MODEL_NAME=modelo,
        EXPERIMENT_SEED=seed,
        EXPERIMENT_NUM_SAMPLES=amostras,
        EXPERIMENT_TASK_CONCURRENCY=tarefas,
        EXPERIMENT_CALL_CONCURRENCY=chamadas,
    )
    try:
        for nome in experimentos:
            output_root = diretorio / "geracao" / nome
            manifesto["experimentos"][nome] = {
                "script": EXPERIMENTOS[nome],
                "output_root": str(output_root),
                "status": "em_andamento",
            }
            if not args.dry_run:
                atualizar_manifesto(caminho_manifesto, manifesto)
            ambiente = ambiente_com_configuracao(
                ambiente_base,
                EXPERIMENT_OUTPUT_ROOT=output_root,
            )
            executar_comando(
                [python_do_projeto(), EXPERIMENTOS[nome]],
                ambiente,
                args.dry_run,
            )
            manifesto["experimentos"][nome]["status"] = "concluido"
            if not args.dry_run:
                atualizar_manifesto(caminho_manifesto, manifesto)
    except Exception:
        manifesto["status"] = "falhou"
        if not args.dry_run:
            atualizar_manifesto(caminho_manifesto, manifesto)
        raise

    manifesto["status"] = "concluido"
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)
        salvar_execucao_atual(diretorio)
    if args.dry_run:
        print("\nSimulacao concluida; nenhum arquivo foi criado.")
    else:
        print(f"\nGeracao concluida: {diretorio}")
        print("Proximo passo: python pipeline_avaliacao.py")


if __name__ == "__main__":
    main()
