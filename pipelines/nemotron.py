"""Pipeline opcional: executa somente o Nemotron sobre a amostra ja congelada."""

import argparse
from datetime import datetime
from pathlib import Path

from .util import (
    ambiente_com_configuracao,
    atualizar_manifesto,
    caminhos_resultados,
    executar_comando,
    ler_manifesto,
    obter_execucao_atual,
    python_do_projeto,
)


def obter_amostra(manifesto, diretorio, dry_run=False):
    amostras = manifesto.get("amostras", {})
    perguntas = Path(amostras.get("perguntas", ""))
    referencias = Path(amostras.get("referencias", ""))
    if perguntas.is_file() and referencias.is_file():
        return perguntas, referencias, False

    perguntas = diretorio / "amostras" / "perguntas_amostradas.json"
    referencias = diretorio / "amostras" / "referencias_amostradas.json"
    if dry_run:
        print("Amostras ausentes: elas seriam reconstruidas dos resultados brutos.")
        return perguntas, referencias, False

    from .geracao import salvar_amostras_canonicas

    manifesto["amostras"] = salvar_amostras_canonicas(
        caminhos_resultados(manifesto),
        diretorio,
        int(manifesto.get("configuracao", {}).get("amostras", 100)),
    )
    return perguntas, referencias, True


def output_root_nemotron(diretorio, experimento_anterior):
    if not experimento_anterior:
        return diretorio / "geracao" / "nemotron_diffusion"
    if experimento_anterior.get("status") == "concluido":
        raise RuntimeError(
            "Esta rodada ja possui uma execucao Nemotron concluida. Crie uma nova "
            "rodada de geracao para manter uma unica resposta Nemotron por pergunta."
        )
    if experimento_anterior.get("status") == "em_andamento":
        raise RuntimeError("Ja existe uma execucao Nemotron marcada como em andamento.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return diretorio / "geracao" / f"nemotron_diffusion_reexecucao_{timestamp}"


def main():
    parser = argparse.ArgumentParser(
        description="Executa somente o Nemotron sobre a amostra da rodada atual."
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
    if manifesto.get("status") != "concluido":
        raise RuntimeError("A geracao principal desta rodada ainda nao foi concluida.")

    perguntas, referencias, amostras_reconstruidas = obter_amostra(
        manifesto,
        diretorio,
        dry_run=args.dry_run,
    )
    anterior = manifesto.get("experimentos", {}).get("nemotron_diffusion")
    output_root = output_root_nemotron(diretorio, anterior)
    esperado = int(manifesto.get("configuracao", {}).get("amostras", 100))
    comando = [
        python_do_projeto(),
        "experimento_nemotron_diffusion.py",
        perguntas,
        "--referencias",
        referencias,
        "--expected-per-dataset",
        str(esperado),
    ]

    if args.dry_run:
        executar_comando(comando, dry_run=True)
        print("\nSimulacao concluida; o Nemotron nao foi carregado.")
        return

    if amostras_reconstruidas:
        atualizar_manifesto(caminho_manifesto, manifesto)

    manifesto.setdefault("experimentos", {})["nemotron_diffusion"] = {
        "script": "experimento_nemotron_diffusion.py",
        "output_root": str(output_root),
        "tipo": "modelo_baseline_externo",
        "amostra_perguntas": str(perguntas),
        "amostra_referencias": str(referencias),
        "status": "em_andamento",
    }
    atualizar_manifesto(caminho_manifesto, manifesto)
    try:
        executar_comando(
            comando,
            ambiente_com_configuracao(EXPERIMENT_OUTPUT_ROOT=output_root),
        )
    except Exception:
        manifesto["experimentos"]["nemotron_diffusion"]["status"] = "falhou"
        atualizar_manifesto(caminho_manifesto, manifesto)
        raise

    manifesto["experimentos"]["nemotron_diffusion"]["status"] = "concluido"
    atualizar_manifesto(caminho_manifesto, manifesto)
    print(f"\nNemotron concluido: {output_root}")
    print("Proximo passo: python -m pipelines.avaliacao")


if __name__ == "__main__":
    main()
