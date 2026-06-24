"""Pipeline 3: consolida metricas e gera os graficos da execucao atual."""

import argparse
from pathlib import Path

from configuracao.ambiente import inteiro, texto
from .util import (
    atualizar_manifesto,
    caminhos_resultados,
    executar_comando,
    ler_manifesto,
    obter_execucao_atual,
    python_do_projeto,
)


def main():
    parser = argparse.ArgumentParser(
        description="Consolida metricas e gera graficos para a execucao definida no .env."
    )
    parser.add_argument("--execucao-dir", help="Sobrescreve PIPELINE_EXECUTION_DIR.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    diretorio = (
        Path(args.execucao_dir).expanduser().resolve()
        if args.execucao_dir
        else obter_execucao_atual()
    )
    caminho_manifesto, manifesto = ler_manifesto(diretorio)
    resultados = caminhos_resultados(manifesto)
    avaliacao = manifesto.get("avaliacao", {})
    diretorio_avaliacao = Path(avaliacao.get("output_root", ""))
    if avaliacao.get("status") != "concluido" or not diretorio_avaliacao.is_dir():
        raise RuntimeError(
            "Avaliacao LLM-as-a-Judge nao encontrada. Execute pipeline_avaliacao.py primeiro."
        )

    seed = inteiro(
        "ANALYSIS_SEED",
        manifesto.get("configuracao", {}).get("seed", 20260612),
    )
    bootstrap = inteiro("BOOTSTRAP_ITERATIONS", 5000, minimo=1)
    output_analise = diretorio / "analise"
    antes = set(output_analise.glob("rodada_*")) if output_analise.exists() else set()
    comando_analise = [
        python_do_projeto(),
        "processar_resultados.py",
        *resultados,
        "--avaliacoes",
        diretorio_avaliacao,
        "--bootstrap-iterations",
        str(bootstrap),
        "--seed",
        str(seed),
        "--output-root",
        output_analise,
    ]
    avaliacao_deterministica = manifesto.get("avaliacao_deterministica", {})
    caminho_deterministico = avaliacao_deterministica.get("output_root")
    diretorio_deterministico = (
        Path(caminho_deterministico) if caminho_deterministico else None
    )
    if diretorio_deterministico is None or not diretorio_deterministico.is_dir():
        raiz_deterministica = diretorio / "avaliacao_deterministica"
        rodadas_deterministicas = (
            sorted(raiz_deterministica.glob("rodada_*"))
            if raiz_deterministica.is_dir()
            else []
        )
        if rodadas_deterministicas:
            diretorio_deterministico = rodadas_deterministicas[-1]
    if (
        diretorio_deterministico is not None and diretorio_deterministico.is_dir()
    ):
        comando_analise.extend(
            ["--avaliacoes-deterministicas", diretorio_deterministico]
        )
    csv_humano = texto("HUMAN_EVALUATION_CSV", None)
    chave_humana = texto("HUMAN_EVALUATION_KEY", None)
    if csv_humano:
        comando_analise.extend(["--avaliacao-humana", csv_humano])
    if chave_humana:
        comando_analise.extend(["--chave-humana", chave_humana])

    manifesto["analise"] = {
        "status": "em_andamento",
        "output_root": str(output_analise),
        "seed": seed,
        "bootstrap_iterations": bootstrap,
    }
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)
    try:
        executar_comando(comando_analise, dry_run=args.dry_run)
        if args.dry_run:
            print("\nAnalise e graficos planejados.")
            return
        rodadas_novas = sorted(set(output_analise.glob("rodada_*")) - antes)
        if len(rodadas_novas) != 1:
            raise RuntimeError("Nao foi possivel identificar a rodada de analise criada.")
        diretorio_metricas = rodadas_novas[0]
        output_graficos = diretorio / "graficos"
        executar_comando(
            [
                python_do_projeto(),
                "gerar_graficos_resultados.py",
                diretorio_metricas,
                "--output-root",
                output_graficos,
            ]
        )
    except Exception:
        manifesto["analise"]["status"] = "falhou"
        if not args.dry_run:
            atualizar_manifesto(caminho_manifesto, manifesto)
        raise

    manifesto["analise"].update(
        {
            "status": "concluido",
            "metricas": str(diretorio_metricas),
            "graficos": str(output_graficos),
        }
    )
    atualizar_manifesto(caminho_manifesto, manifesto)
    print(f"\nMetricas: {diretorio_metricas}")
    print(f"Graficos: {output_graficos}")


if __name__ == "__main__":
    main()
