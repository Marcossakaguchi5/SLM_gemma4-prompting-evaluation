import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


SEED_PADRAO = 20260612


def carregar_checkpoint_jsonl(caminho, chave):
    """Lê um JSONL parcial e preserva o último registro válido por chave.

    Uma interrupção pode deixar apenas a última linha truncada. Ela é ignorada,
    enquanto todos os checkpoints completos anteriores continuam utilizáveis.
    """
    caminho = Path(caminho)
    registros = {}
    if not caminho.is_file():
        return registros

    with caminho.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                item = json.loads(linha)
            except json.JSONDecodeError:
                # Uma linha truncada por desligamento nao invalida checkpoints anteriores.
                continue
            if not isinstance(item, dict):
                continue
            chave_item = chave(item)
            if chave_item is not None:
                registros[chave_item] = item
    return registros


def preparar_diretorio_checkpoint(output_root, variavel_ambiente="EXPERIMENT_RUN_DIR"):
    """Cria uma rodada nova ou reabre a rodada explicitamente informada.

    O pipeline informa ``EXPERIMENT_RUN_DIR`` antes de iniciar cada subprocesso.
    Se esse diretório já existir, trata-se de uma retomada; caso contrário, o
    comportamento independente dos scripts continua criando uma rodada com
    timestamp dentro de ``output_root``.
    """
    configurado = os.environ.get(variavel_ambiente)
    if configurado:
        run_dir = Path(configurado).expanduser()
        if run_dir.exists():
            if not run_dir.is_dir():
                raise NotADirectoryError(f"Diretorio de checkpoint invalido: {run_dir}")
            return run_dir, True
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir, False

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"rodada_{timestamp}"
    sufixo = 1
    while run_dir.exists():
        run_dir = output_root / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1
    run_dir.mkdir()
    return run_dir, False


def chave_checkpoint_resultado(item):
    identificador = item.get("id_instancia")
    abordagem = item.get("abordagem")
    return (identificador, abordagem) if identificador and abordagem else None


def adicionar_jsonl_duravel(caminho, registro):
    """Acrescenta um checkpoint e o descarrega para o disco antes de continuar."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
        arquivo.flush()
        os.fsync(arquivo.fileno())


def salvar_json_atomico(caminho, dados, *, indent=2):
    """Substitui um JSON completo sem deixar um arquivo final parcialmente escrito."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(f".{caminho.name}.tmp")
    with temporario.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=indent)
        arquivo.flush()
        os.fsync(arquivo.fileno())
    os.replace(temporario, caminho)


def seed_experimento():
    return int(os.environ.get("EXPERIMENT_SEED", SEED_PADRAO))


def hash_estavel(texto):
    digest = hashlib.sha256(str(texto).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def extrair_resposta_final(texto):
    texto = "" if texto is None else str(texto)
    padroes = [
        r"RESPOSTA_FINAL\s*:\s*(.+)",
        r"FINAL_ANSWER\s*:\s*(.+)",
        r"FINAL ANSWER\s*:\s*(.+)",
        r"Final answer\s*:\s*(.+)",
        r"Resposta final\s*:\s*(.+)",
    ]
    for padrao in padroes:
        encontrados = re.findall(padrao, texto, flags=re.IGNORECASE)
        if encontrados:
            return encontrados[-1].strip().splitlines()[0].strip()
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    return linhas[-1] if linhas else ""


def _extrair_wrapper_latex(texto, comandos=(r"\boxed{", r"\fbox{", r"\text{")):
    texto = texto.strip()
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


def normalizar_resposta_consenso(texto):
    texto = extrair_resposta_final(texto).strip()
    texto = _extrair_wrapper_latex(texto)
    texto = texto.lower()
    texto = texto.replace("$", "").replace("\\(", "").replace("\\)", "")
    texto = texto.replace("−", "-").replace("–", "-")
    texto = texto.replace(",", "")
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.strip(" .;:")

    alternativa = re.fullmatch(r"(?:option\s*)?\(?([a-e])\)?", texto)
    if alternativa:
        return alternativa.group(1)

    fracao = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", texto)
    if fracao and int(fracao.group(2)) != 0:
        return str(Fraction(int(fracao.group(1)), int(fracao.group(2))))

    try:
        numero = Decimal(texto)
        if numero.is_finite():
            normalizado = format(numero.normalize(), "f")
            return "0" if normalizado in {"-0", "-0.0"} else normalizado
    except InvalidOperation:
        pass
    return texto


def selecionar_trajetoria_consenso(trajetorias, ordem_prioridade):
    candidatos = []
    for chave in ordem_prioridade:
        conteudo = trajetorias[chave]
        resposta_final = extrair_resposta_final(conteudo)
        candidatos.append(
            {
                "chave": chave,
                "resposta_final": resposta_final,
                "resposta_normalizada": normalizar_resposta_consenso(resposta_final),
                "resposta_completa": conteudo,
            }
        )

    contagens = Counter(
        candidato["resposta_normalizada"]
        for candidato in candidatos
        if candidato["resposta_normalizada"]
    )
    maior_contagem = max(contagens.values(), default=0)
    vencedoras = {
        resposta
        for resposta, contagem in contagens.items()
        if contagem == maior_contagem
    }
    selecionada = next(
        (
            candidato
            for candidato in candidatos
            if candidato["resposta_normalizada"] in vencedoras
        ),
        candidatos[0],
    )
    metodo = "majority_vote" if maior_contagem >= 2 else "deterministic_priority_tiebreak"

    resposta_agregada = (
        "GFLOW_SELECTION_WITHOUT_REFERENCE\n"
        f"Selection method: {metodo}.\n"
        f"Selected trajectory: {selecionada['chave']}.\n\n"
        f"{selecionada['resposta_completa']}"
    )
    return resposta_agregada, {
        "metodo": metodo,
        "trajetoria_selecionada": selecionada["chave"],
        "resposta_final_selecionada": selecionada["resposta_final"],
        "contagens_respostas": dict(contagens),
        "candidatos": candidatos,
    }


def amostrar_reprodutivel(itens, quantidade, seed, chave_estrato=None):
    itens = list(itens)
    quantidade = min(quantidade, len(itens))
    if quantidade <= 0:
        return []

    rng = random.Random(seed)
    if not chave_estrato:
        indices = rng.sample(range(len(itens)), quantidade)
        return [(indice, itens[indice]) for indice in indices]

    grupos = defaultdict(list)
    for indice, item in enumerate(itens):
        estrato = chave_estrato(item)
        grupos[str(estrato or "sem_estrato")].append(indice)

    total = len(itens)
    alocacoes = {}
    fracoes = []
    for estrato, indices in grupos.items():
        ideal = quantidade * len(indices) / total
        alocacoes[estrato] = min(len(indices), int(ideal))
        fracoes.append((ideal - int(ideal), estrato))

    faltantes = quantidade - sum(alocacoes.values())
    for _, estrato in sorted(fracoes, key=lambda item: (-item[0], item[1])):
        if faltantes <= 0:
            break
        if alocacoes[estrato] < len(grupos[estrato]):
            alocacoes[estrato] += 1
            faltantes -= 1

    while faltantes > 0:
        houve_alocacao = False
        for estrato in sorted(grupos):
            if alocacoes[estrato] < len(grupos[estrato]):
                alocacoes[estrato] += 1
                faltantes -= 1
                houve_alocacao = True
                if faltantes == 0:
                    break
        if not houve_alocacao:
            break

    selecionados = []
    for estrato in sorted(grupos):
        indices = list(grupos[estrato])
        random.Random(seed + hash_estavel(estrato)).shuffle(indices)
        selecionados.extend(indices[: alocacoes[estrato]])
    rng.shuffle(selecionados)
    return [(indice, itens[indice]) for indice in selecionados]


def _inteiro(valor):
    try:
        return int(valor)
    except (TypeError, ValueError):
        return 0


def _segundos_nanos(valor):
    try:
        return round(float(valor) / 1_000_000_000, 6)
    except (TypeError, ValueError):
        return 0.0


def extrair_telemetria_resposta(resposta):
    usage = getattr(resposta, "usage_metadata", None) or {}
    metadata = getattr(resposta, "response_metadata", None) or {}
    input_tokens = _inteiro(
        usage.get("input_tokens", metadata.get("prompt_eval_count", 0))
    )
    output_tokens = _inteiro(
        usage.get("output_tokens", metadata.get("eval_count", 0))
    )
    total_tokens = _inteiro(usage.get("total_tokens", input_tokens + output_tokens))
    eval_duration = _segundos_nanos(metadata.get("eval_duration"))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "load_duration_seconds": _segundos_nanos(metadata.get("load_duration")),
        "prompt_eval_duration_seconds": _segundos_nanos(
            metadata.get("prompt_eval_duration")
        ),
        "eval_duration_seconds": eval_duration,
        "model_total_duration_seconds": _segundos_nanos(
            metadata.get("total_duration")
        ),
        "tokens_per_second": (
            round(output_tokens / eval_duration, 4)
            if output_tokens and eval_duration
            else None
        ),
    }


def somar_telemetrias(telemetrias):
    campos_soma = [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "load_duration_seconds",
        "prompt_eval_duration_seconds",
        "eval_duration_seconds",
        "model_total_duration_seconds",
    ]
    resultado = {
        campo: round(sum(item.get(campo, 0) or 0 for item in telemetrias), 6)
        for campo in campos_soma
    }
    resultado["input_tokens"] = int(resultado["input_tokens"])
    resultado["output_tokens"] = int(resultado["output_tokens"])
    resultado["total_tokens"] = int(resultado["total_tokens"])
    eval_duration = resultado["eval_duration_seconds"]
    resultado["tokens_per_second"] = (
        round(resultado["output_tokens"] / eval_duration, 4)
        if resultado["output_tokens"] and eval_duration
        else None
    )
    return resultado


def _executar_comando(args, timeout=10):
    try:
        processo = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    texto = (processo.stdout or processo.stderr or "").strip()
    return texto or None


def salvar_manifesto(run_dir, configuracao, ids_amostra, prompts):
    pacotes = {}
    for pacote in (
        "datasets",
        "langchain-core",
        "langchain-ollama",
        "transformers",
        "torch",
        "sympy",
        "psutil",
    ):
        try:
            pacotes[pacote] = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            pacotes[pacote] = None

    modelo = configuracao.get("modelo")
    modelo_descricao = _executar_comando(["ollama", "show", modelo, "--modelfile"]) if modelo else None
    manifesto = {
        "configuracao": configuracao,
        "ids_amostra": ids_amostra,
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "python": sys.version,
        "plataforma": platform.platform(),
        "processador": platform.processor(),
        "cpu_count": os.cpu_count(),
        "pacotes": pacotes,
        "ollama_version": _executar_comando(["ollama", "--version"]),
        "modelo_modelfile_sha256": (
            hashlib.sha256(modelo_descricao.encode("utf-8")).hexdigest()
            if modelo_descricao
            else None
        ),
        "gpu": _executar_comando(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }
    path = Path(run_dir) / "manifesto_execucao.json"
    salvar_json_atomico(path, manifesto)
    return path


class MonitorRecursos:
    def __init__(self, intervalo=0.5):
        self.intervalo = intervalo
        self._parar = threading.Event()
        self._thread = None
        self.metricas = {
            "process_rss_peak_bytes": 0,
            "system_memory_used_peak_bytes": 0,
            "gpu_memory_used_peak_mib": [],
        }

    def _amostrar(self):
        try:
            import psutil

            processo = psutil.Process()
        except ImportError:
            psutil = None
            processo = None

        while not self._parar.is_set():
            if processo is not None:
                try:
                    self.metricas["process_rss_peak_bytes"] = max(
                        self.metricas["process_rss_peak_bytes"],
                        processo.memory_info().rss,
                    )
                    self.metricas["system_memory_used_peak_bytes"] = max(
                        self.metricas["system_memory_used_peak_bytes"],
                        psutil.virtual_memory().used,
                    )
                except (OSError, psutil.Error):
                    pass

            gpu = _executar_comando(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
            )
            if gpu:
                valores = []
                for linha in gpu.splitlines():
                    try:
                        valores.append(float(linha.strip()))
                    except ValueError:
                        valores.append(0.0)
                atuais = self.metricas["gpu_memory_used_peak_mib"]
                if len(atuais) < len(valores):
                    atuais.extend([0.0] * (len(valores) - len(atuais)))
                for indice, valor in enumerate(valores):
                    atuais[indice] = max(atuais[indice], valor)
            self._parar.wait(self.intervalo)

    def iniciar(self):
        self._thread = threading.Thread(target=self._amostrar, daemon=True)
        self._thread.start()
        return self

    def finalizar(self, output_path=None):
        self._parar.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.intervalo * 3))
        self.metricas["monitoring_interval_seconds"] = self.intervalo
        if output_path:
            Path(output_path).write_text(
                json.dumps(self.metricas, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return self.metricas
