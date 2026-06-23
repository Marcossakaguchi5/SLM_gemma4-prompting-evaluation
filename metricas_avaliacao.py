import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from util_experimento import extrair_resposta_final


def _extrair_wrapper_latex(texto, comandos=(r"\boxed{", r"\fbox{", r"\text{")):
    texto = "" if texto is None else str(texto).strip()
    for comando in comandos:
        if not texto.startswith(comando):
            continue
        inicio = len(comando)
        profundidade = 1
        for indice in range(inicio, len(texto)):
            if texto[indice] == "{":
                profundidade += 1
            elif texto[indice] == "}":
                profundidade -= 1
                if profundidade == 0:
                    sufixo = texto[indice + 1 :].strip(" \t\r\n.;:")
                    if not sufixo:
                        return texto[inicio:indice].strip()
                    break
    return texto


def normalizar_texto(texto):
    texto = _extrair_wrapper_latex(texto)
    texto = unicodedata.normalize("NFKC", texto).strip().lower()
    texto = texto.replace("−", "-").replace("–", "-")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip(" \t\r\n.;:")


def extrair_gsm8k_objetivo(gabarito):
    texto = "" if gabarito is None else str(gabarito)
    return texto.split("####")[-1].strip() if "####" in texto else texto.strip()


def extrair_resposta_boxed(solucao):
    if not solucao:
        return None
    texto = str(solucao)
    candidatos = []
    for marcador in (r"\boxed{", r"\fbox{"):
        inicio_busca = 0
        while True:
            inicio = texto.find(marcador, inicio_busca)
            if inicio == -1:
                break
            conteudo_inicio = inicio + len(marcador)
            profundidade = 1
            indice = conteudo_inicio
            while indice < len(texto) and profundidade:
                if texto[indice] == "{":
                    profundidade += 1
                elif texto[indice] == "}":
                    profundidade -= 1
                indice += 1
            if profundidade == 0:
                candidatos.append((inicio, texto[conteudo_inicio : indice - 1].strip()))
            inicio_busca = conteudo_inicio
    return max(candidatos, key=lambda item: item[0])[1] if candidatos else None


def referencia_curta(item):
    if item.get("resposta_boxed"):
        return str(item["resposta_boxed"]).strip()
    dataset = str(item.get("dataset", "")).lower()
    gabarito = item.get("gabarito_oficial", item.get("gabarito", ""))
    if "math" in str(item.get("dataset", "")).lower():
        boxed = extrair_resposta_boxed(gabarito)
        if boxed:
            return boxed
    if "gsm8k" in dataset:
        return extrair_gsm8k_objetivo(gabarito)
    return "" if gabarito is None else str(gabarito).strip()


def exact_match(resposta, referencia):
    return int(normalizar_texto(resposta) == normalizar_texto(referencia))


def _limpar_numero(texto):
    texto = normalizar_texto(texto)
    texto = texto.replace("$", "").replace("%", "")
    texto = texto.replace(",", "")
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", texto)
    return matches[0] if len(matches) == 1 else None


def numeric_match(resposta, referencia, tolerancia=1e-9):
    numero_resposta = _limpar_numero(resposta)
    numero_referencia = _limpar_numero(referencia)
    if numero_resposta is None or numero_referencia is None:
        return None
    try:
        valor_resposta = Decimal(numero_resposta)
        valor_referencia = Decimal(numero_referencia)
    except InvalidOperation:
        return None
    escala = max(abs(valor_referencia), Decimal(1))
    diferenca = abs(valor_resposta - valor_referencia)
    return int(diferenca <= Decimal(str(tolerancia)) * escala)


def _extrair_alternativa(texto):
    texto = normalizar_texto(texto).upper()
    matches = re.findall(r"(?:^|[^A-Z])([A-E])(?:$|[^A-Z])", texto)
    return matches[-1] if matches else None


def multiple_choice_match(resposta, referencia):
    resposta_opcao = _extrair_alternativa(resposta)
    referencia_opcao = _extrair_alternativa(referencia)
    if not resposta_opcao or not referencia_opcao:
        return None
    return int(resposta_opcao == referencia_opcao)


def _limpar_latex(texto):
    texto = normalizar_texto(_extrair_wrapper_latex(texto))
    texto = texto.replace("\\left", "").replace("\\right", "")
    texto = texto.replace("\\,", "").replace("\\!", "")
    texto = texto.strip("$")
    return texto


def _parse_simbolico(texto):
    from sympy import sympify

    texto = _limpar_latex(texto)
    if "\\" in texto or "{" in texto:
        try:
            from sympy.parsing.latex import parse_latex

            return parse_latex(texto)
        except Exception:
            pass

    texto = texto.replace("^", "**")
    texto = texto.replace("\\pi", "pi")
    texto = texto.replace("{", "(").replace("}", ")")
    return sympify(texto)


def symbolic_equivalence(resposta, referencia):
    try:
        from sympy import simplify
    except ImportError:
        return None

    try:
        expressao_resposta = _parse_simbolico(resposta)
        expressao_referencia = _parse_simbolico(referencia)
        return int(simplify(expressao_resposta - expressao_referencia) == 0)
    except Exception:
        return None


def answer_match_objetivo(item, resposta):
    dataset = str(item.get("dataset", "")).lower()
    referencia = referencia_curta(item)
    resultado = {
        "resposta_final": extrair_resposta_final(resposta),
        "referencia_curta": referencia,
        "exact_match": None,
        "numeric_match": None,
        "symbolic_equivalence": None,
        "answer_match_objetivo": None,
        "metrica_objetiva": "",
    }
    candidato = _extrair_wrapper_latex(resultado["resposta_final"])
    resultado["resposta_final"] = candidato
    if not referencia:
        return resultado

    resultado["exact_match"] = exact_match(candidato, referencia)
    if "arc" in dataset:
        resultado["answer_match_objetivo"] = multiple_choice_match(
            candidato,
            referencia,
        )
        resultado["metrica_objetiva"] = "multiple_choice_match"
    elif "gsm8k" in dataset:
        resultado["numeric_match"] = numeric_match(candidato, referencia)
        if resultado["exact_match"] == 1:
            resultado["answer_match_objetivo"] = 1
        else:
            resultado["answer_match_objetivo"] = resultado["numeric_match"]
        resultado["metrica_objetiva"] = "numeric_match"
    elif "math" in dataset or item.get("resposta_boxed"):
        resultado["numeric_match"] = numeric_match(candidato, referencia)
        resultado["symbolic_equivalence"] = (
            1
            if resultado["exact_match"] == 1
            else symbolic_equivalence(candidato, referencia)
        )
        equivalencias = [
            valor
            for valor in (
                resultado["numeric_match"],
                resultado["symbolic_equivalence"],
            )
            if valor is not None
        ]
        if resultado["exact_match"] == 1:
            resultado["answer_match_objetivo"] = 1
        elif equivalencias:
            resultado["answer_match_objetivo"] = max(equivalencias)
        else:
            resultado["answer_match_objetivo"] = None
        resultado["metrica_objetiva"] = "symbolic_equivalence"
    else:
        respostas_corretas = [
            referencia,
            *(item.get("respostas_corretas") or []),
        ]
        normalizado = normalizar_texto(candidato)
        incorretas = {
            normalizar_texto(valor)
            for valor in item.get("respostas_incorretas", [])
            if valor
        }
        correta = any(
            normalizar_texto(valor) == normalizado
            for valor in respostas_corretas
            if valor
        )
        resultado["answer_match_objetivo"] = int(
            correta and normalizado not in incorretas
        )
        resultado["metrica_objetiva"] = "reference_text_match"
    return resultado


def intervalo_bootstrap_binario(valores, iteracoes=5000, seed=20260612, alpha=0.05):
    valores = [int(valor) for valor in valores if valor is not None]
    if not valores:
        return None, None
    import random

    rng = random.Random(seed)
    n = len(valores)
    medias = []
    for _ in range(iteracoes):
        medias.append(sum(valores[rng.randrange(n)] for _ in range(n)) / n)
    medias.sort()
    inferior = medias[max(0, int((alpha / 2) * iteracoes))]
    superior = medias[min(iteracoes - 1, int((1 - alpha / 2) * iteracoes))]
    return round(inferior, 4), round(superior, 4)


def cohens_kappa(valores_a, valores_b):
    pares = [
        (int(a), int(b))
        for a, b in zip(valores_a, valores_b)
        if a is not None and b is not None
    ]
    if not pares:
        return None
    n = len(pares)
    acordo = sum(a == b for a, b in pares) / n
    p_a_1 = sum(a == 1 for a, _ in pares) / n
    p_b_1 = sum(b == 1 for _, b in pares) / n
    esperado = p_a_1 * p_b_1 + (1 - p_a_1) * (1 - p_b_1)
    if math.isclose(esperado, 1.0):
        return 1.0 if math.isclose(acordo, 1.0) else 0.0
    return round((acordo - esperado) / (1 - esperado), 4)


def mcnemar_exato(b, c):
    n = b + c
    if n == 0:
        return 1.0
    menor = min(b, c)
    probabilidade = sum(math.comb(n, k) for k in range(menor + 1)) / (2**n)
    return round(min(1.0, 2 * probabilidade), 6)


def correcao_holm(registros, campo_p="mcnemar_p_exato"):
    validos = [
        (indice, float(registro[campo_p]))
        for indice, registro in enumerate(registros)
        if registro.get(campo_p) not in (None, "")
    ]
    validos.sort(key=lambda item: item[1])
    m = len(validos)
    anterior = 0.0
    for posicao, (indice, p_valor) in enumerate(validos):
        ajustado = min(1.0, (m - posicao) * p_valor)
        ajustado = max(anterior, ajustado)
        registros[indice]["mcnemar_p_holm"] = round(ajustado, 6)
        anterior = ajustado
    return registros
