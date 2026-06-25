"""Funcoes compartilhadas pelos tres pipelines operacionais."""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from configuracao.ambiente import RAIZ_PROJETO, texto


EXPERIMENTOS = {
    "gsm8k_arc": "experimento_gsm8k_arc.py",
    "hendrycks_math": "experimento_hendrycks_math.py",
    "truthfulqa": "experimento_truthfulqa.py",
    "math_avancado": "experimento_math_avancado.py",
}
EXPERIMENTOS_PRINCIPAIS = ("gsm8k_arc", "hendrycks_math", "truthfulqa")


def salvar_json(caminho, dados):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f".{caminho.name}.tmp")
    with temporario.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)
        arquivo.flush()
        os.fsync(arquivo.fileno())
    os.replace(temporario, caminho)


def atualizar_manifesto(caminho, manifesto):
    manifesto["atualizado_em"] = datetime.now().isoformat(timespec="seconds")
    salvar_json(caminho, manifesto)


def executar_comando(comando, ambiente=None, dry_run=False):
    print(f"\n> {subprocess.list2cmdline([str(valor) for valor in comando])}")
    if dry_run:
        return
    processo = subprocess.run(
        comando,
        cwd=RAIZ_PROJETO,
        env=ambiente or os.environ.copy(),
        check=False,
    )
    if processo.returncode:
        raise RuntimeError(
            f"Comando terminou com codigo {processo.returncode}: {comando[1]}"
        )


def criar_execucao():
    raiz_saida = Path(texto("PIPELINE_OUTPUT_ROOT", "resultados")).expanduser()
    if not raiz_saida.is_absolute():
        raiz_saida = RAIZ_PROJETO / raiz_saida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diretorio = raiz_saida / f"rodada_{timestamp}"
    sufixo = 1
    while diretorio.exists():
        diretorio = raiz_saida / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1
    return diretorio


def criar_diretorio_etapa(output_root):
    """Reserva um subdiretório de checkpoint sem criá-lo antecipadamente."""
    output_root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diretorio = output_root / f"rodada_{timestamp}"
    sufixo = 1
    while diretorio.exists():
        diretorio = output_root / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1
    return diretorio


def salvar_execucao_atual(diretorio_execucao):
    caminho = Path(diretorio_execucao).parent / "ultima_execucao.txt"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f".{caminho.name}.tmp")
    with temporario.open("w", encoding="utf-8") as arquivo:
        arquivo.write(str(Path(diretorio_execucao).resolve()))
        arquivo.flush()
        os.fsync(arquivo.fileno())
    os.replace(temporario, caminho)


def obter_execucao_atual():
    configurado = texto("PIPELINE_EXECUTION_DIR", None)
    if configurado:
        diretorio = Path(configurado).expanduser()
        if not diretorio.is_absolute():
            diretorio = RAIZ_PROJETO / diretorio
        return diretorio.resolve()

    raiz_saida = Path(texto("PIPELINE_OUTPUT_ROOT", "resultados")).expanduser()
    if not raiz_saida.is_absolute():
        raiz_saida = RAIZ_PROJETO / raiz_saida
    apontador = raiz_saida / "ultima_execucao.txt"
    if not apontador.exists():
        raise FileNotFoundError(
            "Nenhuma execucao atual foi encontrada. Execute primeiro pipeline_geracao.py."
        )
    return Path(apontador.read_text(encoding="utf-8").strip()).resolve()


def ler_manifesto(diretorio_execucao):
    caminho = Path(diretorio_execucao) / "manifesto_pipeline.json"
    if not caminho.exists():
        raise FileNotFoundError(f"Manifesto nao encontrado em {caminho}.")
    with caminho.open("r", encoding="utf-8") as arquivo:
        return caminho, json.load(arquivo)


def caminhos_resultados(manifesto):
    caminhos = []
    for nome, dados in manifesto.get("experimentos", {}).items():
        if dados.get("status") != "concluido":
            continue
        caminho = Path(dados["output_root"])
        if not caminho.is_dir():
            raise FileNotFoundError(
                f"Resultados de {nome} nao encontrados em {caminho}."
            )
        caminhos.append(caminho)
    if not caminhos:
        raise RuntimeError("O manifesto nao possui nenhum experimento concluido.")
    return caminhos


def ambiente_com_configuracao(base=None, **variaveis):
    ambiente = (base or os.environ).copy()
    ambiente.update({chave: str(valor) for chave, valor in variaveis.items()})
    return ambiente


def python_do_projeto():
    return sys.executable
