"""Pipeline 2: avalia a execucao atual com LLM-as-a-Judge."""

import argparse
from datetime import datetime
from pathlib import Path

from configuracao.ambiente import booleano, decimal, inteiro, texto
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
        description="Executa o LLM-as-a-Judge para a execucao definida no .env."
    )
    parser.add_argument("--execucao-dir", help="Sobrescreve PIPELINE_EXECUTION_DIR.")
    parser.add_argument(
        "--refazer",
        action="store_true",
        help="Cria uma nova avaliação para a rodada, preservando a avaliação anterior.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    diretorio = (
        Path(args.execucao_dir).expanduser().resolve()
        if args.execucao_dir
        else obter_execucao_atual()
    )
    caminho_manifesto, manifesto = ler_manifesto(diretorio)
    if manifesto.get("status") != "concluido":
        raise RuntimeError(
            "A geracao desta rodada ainda nao foi concluida. Aguarde todos os benchmarks terminarem."
        )
    avaliacao_anterior = manifesto.get("avaliacao", {})
    if avaliacao_anterior.get("status") == "concluido" and not args.refazer:
        raise RuntimeError(
            "Esta execucao ja possui uma avaliacao concluida. Para avaliar novamente "
            "as mesmas respostas, execute com --refazer."
        )

    resultados = caminhos_resultados(manifesto)
    provider = texto("JUDGE_PROVIDER", "ollama")
    if provider not in {"ollama", "gemini", "openrouter"}:
        raise ValueError("JUDGE_PROVIDER deve ser ollama, gemini ou openrouter.")
    modelo_juiz = texto("JUDGE_MODEL_NAME")
    if not modelo_juiz:
        raise ValueError("Defina JUDGE_MODEL_NAME no .env.")
    seed = inteiro(
        "JUDGE_SEED",
        manifesto.get("configuracao", {}).get("seed", 20260612),
    )
    amostra_secundaria = decimal("SECONDARY_JUDGE_SAMPLE_RATE", 0.10, 0, 1)
    if args.refazer and avaliacao_anterior.get("status") == "concluido":
        historico = manifesto.setdefault("historico_avaliacoes", [])
        historico.append(avaliacao_anterior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = diretorio / f"avaliacao_juiz_reavaliacao_{timestamp}"
    else:
        output_root = diretorio / "avaliacao_juiz"
    output_deterministico = diretorio / "avaliacao_deterministica"

    manifesto["avaliacao_deterministica"] = {
        "status": "em_andamento",
        "output_root": str(output_deterministico),
    }
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)
    try:
        executar_comando(
            [
                python_do_projeto(),
                "avaliar_deterministico.py",
                *resultados,
                "--output-root",
                output_deterministico,
            ],
            dry_run=args.dry_run,
        )
    except Exception:
        manifesto["avaliacao_deterministica"]["status"] = "falhou"
        if not args.dry_run:
            atualizar_manifesto(caminho_manifesto, manifesto)
        raise
    manifesto["avaliacao_deterministica"]["status"] = "concluido"
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)

    comando = [
        python_do_projeto(),
        "avaliar_llm_judge.py",
        *resultados,
        "--provider",
        provider,
        "--judge-model",
        modelo_juiz,
        "--output-root",
        output_root,
        "--seed",
        str(seed),
        "--secondary-sample-rate",
        str(amostra_secundaria),
        "--sleep",
        str(decimal("JUDGE_SLEEP_SECONDS", 0, 0)),
    ]
    modelo_secundario = texto("SECONDARY_JUDGE_MODEL")
    if modelo_secundario:
        provider_secundario = texto("SECONDARY_JUDGE_PROVIDER", "ollama")
        if provider_secundario not in {"ollama", "gemini", "openrouter"}:
            raise ValueError(
                "SECONDARY_JUDGE_PROVIDER deve ser ollama, gemini ou openrouter."
            )
        comando.extend(["--secondary-provider", provider_secundario])
        comando.extend(["--secondary-judge-model", modelo_secundario])
    limite = texto("JUDGE_LIMIT", None)
    if limite is not None:
        comando.extend(["--limite", str(inteiro("JUDGE_LIMIT", 0, 1))])
    if booleano("JUDGE_ALLOW_SAME_MODEL", False):
        comando.append("--permitir-mesmo-modelo")
    if not booleano("JUDGE_POSITIONAL_AUDIT", True):
        comando.append("--sem-auditoria-posicional")
    if booleano("JUDGE_ONLY_DETERMINISTIC_FALLBACK", True):
        comando.append("--apenas-fallback-deterministico")

    manifesto["avaliacao"] = {
        "status": "em_andamento",
        "output_root": str(output_root),
        "provider": provider,
        "judge_model": modelo_juiz,
        "seed": seed,
    }
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)
    try:
        executar_comando(comando, dry_run=args.dry_run)
    except Exception:
        manifesto["avaliacao"]["status"] = "falhou"
        if not args.dry_run:
            atualizar_manifesto(caminho_manifesto, manifesto)
        raise

    manifesto["avaliacao"]["status"] = "concluido"
    if not args.dry_run:
        atualizar_manifesto(caminho_manifesto, manifesto)
    if args.dry_run:
        print("\nSimulacao concluida; nenhum arquivo foi criado nem houve chamada ao juiz.")
    else:
        print(f"\nAvaliacao concluida: {output_root}")
        print("Proximo passo: python -m pipelines.graficos")


if __name__ == "__main__":
    main()
