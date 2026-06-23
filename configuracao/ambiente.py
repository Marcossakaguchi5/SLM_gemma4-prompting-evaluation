"""Carregamento centralizado da configuracao local definida em .env."""

import os
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parent.parent
ARQUIVO_ENV = RAIZ_PROJETO / ".env"


def carregar_ambiente(caminho=ARQUIVO_ENV):
    """Carrega pares CHAVE=VALOR simples, sem expor valores sensiveis em logs."""
    caminho = Path(caminho)
    if not caminho.exists():
        return

    for numero_linha, linha in enumerate(
        caminho.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        if linha.startswith("export "):
            linha = linha[7:].lstrip()
        if "=" not in linha:
            raise ValueError(f"Linha invalida em {caminho.name}:{numero_linha}.")
        chave, valor = linha.split("=", 1)
        chave = chave.strip()
        valor = valor.strip()
        if not chave or not chave.replace("_", "").isalnum() or chave[0].isdigit():
            raise ValueError(f"Chave invalida em {caminho.name}:{numero_linha}.")
        if len(valor) >= 2 and valor[0] == valor[-1] and valor[0] in {"'", '"'}:
            valor = valor[1:-1]
        os.environ[chave] = valor


def texto(chave, padrao=None):
    valor = os.environ.get(chave, padrao)
    if valor is None:
        return None
    valor = str(valor).strip()
    return valor if valor else padrao


def inteiro(chave, padrao, minimo=None):
    valor = texto(chave, str(padrao))
    try:
        numero = int(valor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{chave} deve ser um inteiro; recebido: {valor!r}.") from exc
    if minimo is not None and numero < minimo:
        raise ValueError(f"{chave} deve ser maior ou igual a {minimo}.")
    return numero


def decimal(chave, padrao, minimo=None, maximo=None):
    valor = texto(chave, str(padrao))
    try:
        numero = float(valor)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{chave} deve ser numerico; recebido: {valor!r}.") from exc
    if minimo is not None and numero < minimo:
        raise ValueError(f"{chave} deve ser maior ou igual a {minimo}.")
    if maximo is not None and numero > maximo:
        raise ValueError(f"{chave} deve ser menor ou igual a {maximo}.")
    return numero


def booleano(chave, padrao=False):
    valor = texto(chave, "true" if padrao else "false").lower()
    if valor in {"1", "true", "yes", "sim", "on"}:
        return True
    if valor in {"0", "false", "no", "nao", "off"}:
        return False
    raise ValueError(f"{chave} deve ser true ou false; recebido: {valor!r}.")


def lista(chave, padrao=()):
    valor = texto(chave, None)
    if valor is None:
        return tuple(padrao)
    return tuple(item.strip() for item in valor.split(",") if item.strip())


carregar_ambiente()
