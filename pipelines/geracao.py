"""Pipeline 1: gera os dados brutos dos experimentos configurados no .env."""

import argparse
from datetime import datetime
from pathlib import Path

from collections import Counter

from configuracao.ambiente import inteiro, lista, texto
from configuracao.prompts import PROMPT_VERSION
from .util import (
    EXPERIMENTOS,
    EXPERIMENTOS_PRINCIPAIS,
    ambiente_com_configuracao,
    atualizar_manifesto,
    criar_diretorio_etapa,
    criar_execucao,
    executar_comando,
    ler_manifesto,
    obter_execucao_atual,
    python_do_projeto,
    salvar_json,
    salvar_execucao_atual,
)


def obter_experimentos(valor_cli=None):
    if valor_cli:
        selecionados = tuple(
            dict.fromkeys(item.strip() for item in valor_cli.split(",") if item.strip())
        )
    else:
        selecionados = tuple(
            dict.fromkeys(lista("PIPELINE_EXPERIMENTS", EXPERIMENTOS_PRINCIPAIS))
        )
    desconhecidos = set(selecionados) - set(EXPERIMENTOS)
    if desconhecidos:
        raise ValueError(
            "PIPELINE_EXPERIMENTS contem experimentos invalidos: "
            f"{', '.join(sorted(desconhecidos))}."
        )
    if not selecionados:
        raise ValueError("A selecao de experimentos nao pode ficar vazia.")
    return selecionados


def salvar_amostras_canonicas(fontes, diretorio, esperado_por_dataset):
    """Separa perguntas de referencias para reutilizar a mesma amostra em outro modelo."""
    from experimento_nemotron_diffusion import (
        extrair_instancias_canonicas,
        validar_tamanho_amostra,
    )

    instancias = extrair_instancias_canonicas(fontes)
    validar_tamanho_amostra(instancias, esperado_por_dataset)
    pasta_amostras = diretorio / "amostras"
    pasta_amostras.mkdir(parents=True, exist_ok=True)
    perguntas_path = pasta_amostras / "perguntas_amostradas.json"
    referencias_path = pasta_amostras / "referencias_amostradas.json"

    campos_contexto = (
        "indice_original",
        "seed_amostragem",
        "subset",
        "level",
        "config",
        "split",
        "categoria",
        "tipo",
        "fonte",
    )
    campos_referencia = (
        "gabarito_oficial",
        "gabarito",
        "resposta_boxed",
        "respostas_corretas",
        "respostas_incorretas",
    )
    perguntas = []
    referencias = []
    for item in instancias:
        base = {
            "id_instancia": item["id_instancia"],
            "dataset": item.get("dataset"),
        }
        perguntas.append(
            {
                **base,
                "pergunta": item.get("pergunta"),
                **{campo: item[campo] for campo in campos_contexto if campo in item},
            }
        )
        referencias.append(
            {
                **base,
                **{
                    campo: item[campo]
                    for campo in campos_referencia
                    if campo in item
                },
            }
        )
    salvar_json(perguntas_path, perguntas)
    salvar_json(referencias_path, referencias)
    return {
        "status": "concluido",
        "perguntas": str(perguntas_path),
        "referencias": str(referencias_path),
        "total": len(perguntas),
        "contagem_por_dataset": dict(
            sorted(Counter(item.get("dataset", "") for item in perguntas).items())
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Gera os dados dos experimentos definidos no .env."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retomar",
        action="store_true",
        help="Retoma uma rodada de geracao interrompida usando os JSONL parciais.",
    )
    parser.add_argument("--execucao-dir", help="Rodada a retomar.")
    parser.add_argument(
        "--experimentos",
        help=(
            "Sobrescreve PIPELINE_EXPERIMENTS para esta execucao. "
            "Exemplo: gsm8k_arc,hendrycks_math"
        ),
    )
    args = parser.parse_args()

    if args.retomar:
        diretorio = (
            Path(args.execucao_dir).expanduser().resolve()
            if args.execucao_dir
            else obter_execucao_atual()
        )
        caminho_manifesto, manifesto = ler_manifesto(diretorio)
        if manifesto.get("tipo") != "pipeline_geracao":
            raise RuntimeError("O diretorio informado nao pertence ao pipeline de geracao.")
        if manifesto.get("status") == "concluido":
            raise RuntimeError("Esta rodada de geracao ja esta concluida; nao ha checkpoint pendente.")
        configuracao = manifesto.get("configuracao", {})
        amostras = int(configuracao.get("amostras", 100))
        seed = int(configuracao.get("seed", 20260612))
        modelo = configuracao.get("modelo", "gemma4:e4b")
        tarefas = int(configuracao.get("task_concurrency", 16))
        chamadas = int(configuracao.get("call_concurrency", 4))
        experimentos = tuple(
            nome for nome in manifesto.get("experimentos", {}) if nome in EXPERIMENTOS
        )
        if not experimentos:
            raise RuntimeError("O manifesto nao possui benchmarks validos para retomar.")
        print(f"Retomando execucao: {diretorio}")
    else:
        amostras = inteiro("EXPERIMENT_NUM_SAMPLES", 100, minimo=1)
        seed = inteiro("EXPERIMENT_SEED", 20260612)
        modelo = texto("SLM_MODEL_NAME", "gemma4:e4b")
        tarefas = inteiro("EXPERIMENT_TASK_CONCURRENCY", 16, minimo=1)
        chamadas = inteiro("EXPERIMENT_CALL_CONCURRENCY", 4, minimo=1)
        experimentos = obter_experimentos(args.experimentos)
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
            # Permite recuperar a rodada mesmo se o computador desligar logo depois.
            salvar_execucao_atual(diretorio)

    ambiente_base = ambiente_com_configuracao(
        SLM_MODEL_NAME=modelo,
        EXPERIMENT_SEED=seed,
        EXPERIMENT_NUM_SAMPLES=amostras,
        EXPERIMENT_TASK_CONCURRENCY=tarefas,
        EXPERIMENT_CALL_CONCURRENCY=chamadas,
    )
    try:
        fontes_amostra = []
        for nome in experimentos:
            anterior = manifesto["experimentos"].get(nome, {})
            output_root = Path(anterior.get("output_root", diretorio / "geracao" / nome))
            if anterior.get("status") == "concluido":
                fontes_amostra.append(output_root)
                continue
            run_dir_anterior = anterior.get("run_dir")
            if args.retomar and not run_dir_anterior:
                raise RuntimeError(
                    f"O checkpoint de {nome} foi criado por uma versao anterior sem run_dir. "
                    "Inicie uma nova rodada para usar a retomada segura."
                )
            run_dir = (
                Path(run_dir_anterior)
                if run_dir_anterior
                else criar_diretorio_etapa(output_root)
            )
            manifesto["experimentos"][nome] = {
                **anterior,
                "script": EXPERIMENTOS[nome],
                "output_root": str(output_root),
                "run_dir": str(run_dir),
                "status": "em_andamento",
            }
            if not args.dry_run:
                atualizar_manifesto(caminho_manifesto, manifesto)
            ambiente = ambiente_com_configuracao(
                ambiente_base,
                EXPERIMENT_OUTPUT_ROOT=output_root,
                EXPERIMENT_RUN_DIR=run_dir,
            )
            executar_comando(
                [python_do_projeto(), EXPERIMENTOS[nome]],
                ambiente,
                args.dry_run,
            )
            fontes_amostra.append(output_root)
            manifesto["experimentos"][nome]["status"] = "concluido"
            if not args.dry_run:
                atualizar_manifesto(caminho_manifesto, manifesto)

        if args.dry_run:
            print(
                "\nAmostras planejadas: amostras/perguntas_amostradas.json e "
                "amostras/referencias_amostradas.json"
            )
        else:
            amostras_anteriores = manifesto.get("amostras", {})
            perguntas = Path(amostras_anteriores.get("perguntas", ""))
            referencias = Path(amostras_anteriores.get("referencias", ""))
            if not perguntas.is_file() or not referencias.is_file():
                manifesto["amostras"] = salvar_amostras_canonicas(
                    fontes_amostra,
                    diretorio,
                    amostras,
                )
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
        print("Proximo passo: python -m pipelines.nemotron (opcional) ou python -m pipelines.avaliacao")


if __name__ == "__main__":
    main()
