"""Pipeline complementar: nova rodada objetiva, sem TruthfulQA.

Este modulo e um orquestrador fino sobre os pipelines existentes. Ele nao
duplica a logica de geracao, avaliacao ou graficos; apenas fixa a selecao de
benchmarks para datasets objetivos e garante que avaliacao/graficos usem a
rodada recem-criada.
"""

import argparse
from pathlib import Path

from configuracao.ambiente import lista
from .util import (
    atualizar_manifesto,
    executar_comando,
    ler_manifesto,
    obter_execucao_atual,
    python_do_projeto,
)


EXPERIMENTOS_OBJETIVOS_PADRAO = ("gsm8k_arc", "hendrycks_math")
EXPERIMENTOS_PERMITIDOS = set(EXPERIMENTOS_OBJETIVOS_PADRAO)


def obter_experimentos_objetivos():
    selecionados = tuple(
        dict.fromkeys(
            lista("PIPELINE_CONTINUACAO_EXPERIMENTS", EXPERIMENTOS_OBJETIVOS_PADRAO)
        )
    )
    invalidos = set(selecionados) - EXPERIMENTOS_PERMITIDOS
    if invalidos:
        raise ValueError(
            "PIPELINE_CONTINUACAO_EXPERIMENTS aceita apenas datasets objetivos "
            "deste fluxo complementar: "
            f"{', '.join(EXPERIMENTOS_OBJETIVOS_PADRAO)}. "
            f"Recebido invalido: {', '.join(sorted(invalidos))}."
        )
    if not selecionados:
        raise ValueError("PIPELINE_CONTINUACAO_EXPERIMENTS nao pode ficar vazio.")
    return selecionados


def registrar_perfil(diretorio, experimentos, dry_run=False):
    if dry_run:
        return
    caminho_manifesto, manifesto = ler_manifesto(diretorio)
    manifesto["perfil_pipeline"] = {
        "nome": "continuacao_sem_truthful",
        "descricao": (
            "Rodada complementar apenas com datasets objetivos; TruthfulQA "
            "nao e gerado nem avaliado nesta execucao."
        ),
        "experimentos": list(experimentos),
    }
    atualizar_manifesto(caminho_manifesto, manifesto)


def comando_modulo(modulo, *argumentos):
    return [python_do_projeto(), "-m", modulo, *map(str, argumentos)]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cria uma continuacao experimental sem TruthfulQA: gera apenas "
            "datasets objetivos, avalia a rodada criada e produz graficos."
        )
    )
    parser.add_argument(
        "--execucao-dir",
        help=(
            "Rodada existente para retomar/avaliar. Se omitido, a geracao cria "
            "uma nova rodada e as etapas seguintes usam essa rodada."
        ),
    )
    parser.add_argument(
        "--retomar-geracao",
        action="store_true",
        help="Retoma a geracao objetiva interrompida na rodada informada/atual.",
    )
    parser.add_argument(
        "--pular-geracao",
        action="store_true",
        help="Nao gera novas respostas; usa --execucao-dir ou a ultima execucao.",
    )
    parser.add_argument(
        "--sem-avaliacao",
        action="store_true",
        help="Para depois da geracao, sem chamar o LLM-juiz/fallback.",
    )
    parser.add_argument(
        "--retomar-avaliacao",
        action="store_true",
        help="Retoma uma avaliacao interrompida para esta rodada.",
    )
    parser.add_argument(
        "--refazer-avaliacao",
        action="store_true",
        help="Reavalia a rodada preservando uma avaliacao anterior ja concluida.",
    )
    parser.add_argument(
        "--sem-graficos",
        action="store_true",
        help="Para depois da avaliacao, sem consolidar metricas/graficos.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.pular_geracao and args.retomar_geracao:
        parser.error("Use apenas um entre --pular-geracao e --retomar-geracao.")
    if args.refazer_avaliacao and args.retomar_avaliacao:
        parser.error("Use apenas um entre --refazer-avaliacao e --retomar-avaliacao.")

    experimentos = obter_experimentos_objetivos()
    valor_experimentos = ",".join(experimentos)

    diretorio = (
        Path(args.execucao_dir).expanduser().resolve()
        if args.execucao_dir
        else None
    )

    if not args.pular_geracao:
        comando_geracao = comando_modulo(
            "pipelines.geracao",
            "--experimentos",
            valor_experimentos,
        )
        if args.retomar_geracao:
            comando_geracao.append("--retomar")
            if diretorio is not None:
                comando_geracao.extend(["--execucao-dir", diretorio])
        elif diretorio is not None:
            raise RuntimeError(
                "Para usar --execucao-dir na etapa de geracao, combine com "
                "--retomar-geracao. Para avaliar/grafar uma rodada ja pronta, "
                "use --pular-geracao."
            )
        if args.dry_run:
            comando_geracao.append("--dry-run")
        executar_comando(comando_geracao, dry_run=False)
        if args.dry_run and diretorio is None and not args.retomar_geracao:
            print(
                "\nDry-run: a geracao objetiva criaria uma nova rodada. "
                "Avaliacao e graficos dependem do diretorio real criado."
            )
            return
        diretorio = obter_execucao_atual() if diretorio is None else diretorio
        registrar_perfil(diretorio, experimentos, args.dry_run)

    if diretorio is None:
        diretorio = obter_execucao_atual()

    if args.sem_avaliacao:
        print(f"\nContinuacao objetiva gerada em: {diretorio}")
        print(
            "Proximo passo: python -m pipelines.continuacao_sem_truthful "
            f"--pular-geracao --execucao-dir {diretorio}"
        )
        return

    comando_avaliacao = comando_modulo(
        "pipelines.avaliacao",
        "--execucao-dir",
        diretorio,
    )
    if args.retomar_avaliacao:
        comando_avaliacao.append("--retomar")
    if args.refazer_avaliacao:
        comando_avaliacao.append("--refazer")
    if args.dry_run:
        comando_avaliacao.append("--dry-run")
    executar_comando(comando_avaliacao, dry_run=False)

    if args.sem_graficos:
        print(f"\nContinuacao objetiva avaliada em: {diretorio}")
        print(f"Proximo passo: python -m pipelines.graficos --execucao-dir {diretorio}")
        return

    comando_graficos = comando_modulo(
        "pipelines.graficos",
        "--execucao-dir",
        diretorio,
    )
    if args.dry_run:
        comando_graficos.append("--dry-run")
    executar_comando(comando_graficos, dry_run=False)

    print(f"\nContinuacao objetiva concluida: {diretorio}")


if __name__ == "__main__":
    main()
