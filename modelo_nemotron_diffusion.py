import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "nvidia/Nemotron-Labs-Diffusion-3B"
MODOS_VALIDOS = ("diffusion", "ar", "linear-spec")
DEFAULT_PERGUNTA = (
    "What is the smallest value of $x$ such that $|5x - 1| = |3x + 2|$? "
    "Express your answer as a common fraction"
)


def selecionar_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def selecionar_dtype(dtype: str, device: torch.device):
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type != "cuda":
        return "auto"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def carregar_modelo(model_id: str = MODEL_ID, device: str = "auto", dtype: str = "auto"):
    torch_device = selecionar_device(device)
    torch_dtype = selecionar_dtype(dtype, torch_device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(torch_device).eval()
    return tokenizer, model, torch_device


def montar_prompt(
    tokenizer,
    pergunta: str,
    instrucao_sistema: Optional[str] = None,
) -> torch.Tensor:
    """Monta uma conversa no formato nativo do tokenizer do Nemotron."""
    messages = []
    if instrucao_sistema:
        messages.append({"role": "system", "content": instrucao_sistema})
    messages.append({"role": "user", "content": pergunta})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt, return_tensors="pt").input_ids


def normalizar_nfe(nfe):
    """Converte o contador opcional do modelo para um valor serializavel."""
    try:
        return int(nfe)
    except (TypeError, ValueError):
        return nfe


def ajustar_tokens_diffusion(max_new_tokens: int, block_length: int) -> int:
    if max_new_tokens % block_length == 0:
        return max_new_tokens

    ajustado = math.ceil(max_new_tokens / block_length) * block_length
    print(
        (
            f"Aviso: em modo diffusion, max_new_tokens precisa ser multiplo de "
            f"block_length. Ajustando {max_new_tokens} para {ajustado}."
        ),
        file=sys.stderr,
    )
    return ajustado


def gerar_ids(
    model,
    prompt_ids: torch.Tensor,
    modo: str,
    max_new_tokens: int,
    block_length: int,
    threshold: float,
    temperature: float,
    eos_token_id: int,
):
    if modo == "ar":
        return model.ar_generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )

    if modo == "linear-spec":
        return model.linear_spec_generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            block_length=block_length,
            threshold=threshold,
            temperature=temperature,
            eos_token_id=eos_token_id,
        )

    max_new_tokens = ajustar_tokens_diffusion(max_new_tokens, block_length)
    return model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        threshold=threshold,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )


class NemotronDiffusionRunner:
    """Mantem o Nemotron carregado para responder varias perguntas em sequencia.

    O script original carregava o modelo a cada chamada de ``gerar_resposta``.
    Para benchmarks isso multiplicaria o tempo de carregamento por centenas de
    itens. Esta classe carrega uma unica vez e expoe uma inferencia por item.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        modo: str = "diffusion",
        max_new_tokens: int = 64,
        block_length: int = 32,
        threshold: float = 0.9,
        temperature: float = 0.0,
        device: str = "auto",
        dtype: str = "auto",
    ):
        if modo not in MODOS_VALIDOS:
            raise ValueError(f"Modo invalido: {modo}. Use um de {MODOS_VALIDOS}.")

        self.model_id = model_id
        self.modo = modo
        self.max_new_tokens = max_new_tokens
        self.block_length = block_length
        self.threshold = threshold
        self.temperature = temperature
        self.dtype = dtype
        self.tokenizer, self.model, self.torch_device = carregar_modelo(
            model_id=model_id,
            device=device,
            dtype=dtype,
        )

    def gerar(
        self,
        pergunta: str,
        instrucao_sistema: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """Gera uma resposta e telemetria padronizada para um item."""
        prompt_ids = montar_prompt(
            self.tokenizer,
            pergunta,
            instrucao_sistema=instrucao_sistema,
        ).to(self.torch_device)
        inicio = time.perf_counter()
        with torch.inference_mode():
            out_ids, nfe = gerar_ids(
                model=self.model,
                prompt_ids=prompt_ids,
                modo=self.modo,
                max_new_tokens=self.max_new_tokens,
                block_length=self.block_length,
                threshold=self.threshold,
                temperature=self.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        duracao = time.perf_counter() - inicio
        resposta_ids = out_ids[:, prompt_ids.shape[1] :]
        texto = self.tokenizer.batch_decode(
            resposta_ids,
            skip_special_tokens=True,
        )[0].strip()
        input_tokens = int(prompt_ids.shape[1])
        output_tokens = int(resposta_ids.shape[1])
        return texto, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model_total_duration_seconds": round(duracao, 4),
            "tokens_per_second": (
                round(output_tokens / duracao, 4) if output_tokens and duracao else None
            ),
            "nfe": normalizar_nfe(nfe),
            "device": str(self.torch_device),
            "dtype": self.dtype,
            "mode": self.modo,
        }


def gerar_resposta(
    pergunta: str,
    modo: str = "diffusion",
    max_new_tokens: int = 64,
    block_length: int = 32,
    threshold: float = 0.9,
    temperature: float = 0.0,
    device: str = "auto",
    dtype: str = "auto",
) -> Tuple[str, int]:
    runner = NemotronDiffusionRunner(
        modo=modo,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        threshold=threshold,
        temperature=temperature,
        device=device,
        dtype=dtype,
    )
    texto, telemetria = runner.gerar(pergunta)
    return texto, telemetria["nfe"]


def executar_com_metricas(args) -> Dict:
    inicio_total = time.perf_counter()
    inicio_carregamento = time.perf_counter()
    runner = NemotronDiffusionRunner(
        modo=args.mode,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        threshold=args.threshold,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype,
    )
    fim_carregamento = time.perf_counter()
    resposta, telemetria = runner.gerar(args.pergunta)
    fim_total = time.perf_counter()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": runner.model_id,
        "pergunta": args.pergunta,
        "resposta": resposta,
        "tempo_total_segundos": round(fim_total - inicio_total, 4),
        "tempo_carregamento_segundos": round(fim_carregamento - inicio_carregamento, 4),
        "tempo_geracao_segundos": telemetria["model_total_duration_seconds"],
        "nfe": telemetria["nfe"],
        "parametros": {
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "threshold": args.threshold,
            "temperature": args.temperature,
            "device": str(runner.torch_device),
            "dtype": args.dtype,
        },
    }


def salvar_json(resultado: Dict, output: str):
    caminho = Path(output)
    if caminho.parent != Path("."):
        caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa o modelo nvidia/Nemotron-Labs-Diffusion-3B localmente."
    )
    parser.add_argument(
        "pergunta",
        nargs="?",
        default=DEFAULT_PERGUNTA,
        help="Mensagem enviada ao modelo.",
    )
    parser.add_argument(
        "--mode",
        choices=MODOS_VALIDOS,
        default="diffusion",
        help="Modo de geracao do modelo.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Quantidade maxima de tokens novos gerados.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Tamanho do bloco usado nos modos diffusion e linear-spec.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Limiar usado nos modos diffusion e linear-spec.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de amostragem.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Dispositivo: auto, cuda, cuda:0 ou cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Precisao do modelo.",
    )
    parser.add_argument(
        "--show-nfe",
        action="store_true",
        help="Mostra o numero de avaliacoes de funcao reportado pelo modelo.",
    )
    parser.add_argument(
        "--output",
        default="resultado_nemotron_diffusion.json",
        help="Arquivo JSON onde o resultado sera salvo.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    resultado = executar_com_metricas(args)
    salvar_json(resultado, args.output)

    print(resultado["resposta"])
    print(f"Tempo total: {resultado['tempo_total_segundos']}s")
    print(f"Tempo de geracao: {resultado['tempo_geracao_segundos']}s")
    print(f"JSON salvo em: {args.output}")
    if args.show_nfe:
        print(f"[NFE={resultado['nfe']}]")


if __name__ == "__main__":
    main()
