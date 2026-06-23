"""Pipeline 2: avalia a execucao atual com LLM-as-a-Judge."""

import argparse
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
    if manifesto.get("avaliacao", {}).get("status") == "concluido":
        raise RuntimeError(
            "Esta execucao ja possui uma avaliacao concluida; use outra execucao para "
            "nao misturar rodadas de julgamento."
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
    output_root = diretorio / "avaliacao_juiz"

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
    print(f"\nAvaliacao concluida: {output_root}")
    print("Proximo passo: python pipeline_graficos.py")


if __name__ == "__main__":
    main()
