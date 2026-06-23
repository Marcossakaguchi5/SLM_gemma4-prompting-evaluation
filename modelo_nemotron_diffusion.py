import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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


def montar_prompt(tokenizer, pergunta: str) -> torch.Tensor:
    messages = [{"role": "user", "content": pergunta}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt, return_tensors="pt").input_ids


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
    tokenizer, model, torch_device = carregar_modelo(device=device, dtype=dtype)
    prompt_ids = montar_prompt(tokenizer, pergunta).to(torch_device)

    out_ids, nfe = gerar_ids(
        model=model,
        prompt_ids=prompt_ids,
        modo=modo,
        max_new_tokens=max_new_tokens,
        block_length=block_length,
        threshold=threshold,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    texto = tokenizer.batch_decode(
        out_ids[:, prompt_ids.shape[1] :],
        skip_special_tokens=True,
    )[0]
    return texto.strip(), nfe


def executar_com_metricas(args) -> Dict:
    inicio_total = time.perf_counter()
    inicio_carregamento = time.perf_counter()
    tokenizer, model, torch_device = carregar_modelo(device=args.device, dtype=args.dtype)
    fim_carregamento = time.perf_counter()

    prompt_ids = montar_prompt(tokenizer, args.pergunta).to(torch_device)

    inicio_geracao = time.perf_counter()
    out_ids, nfe = gerar_ids(
        model=model,
        prompt_ids=prompt_ids,
        modo=args.mode,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        threshold=args.threshold,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    fim_geracao = time.perf_counter()

    resposta = tokenizer.batch_decode(
        out_ids[:, prompt_ids.shape[1] :],
        skip_special_tokens=True,
    )[0].strip()
    fim_total = time.perf_counter()

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "pergunta": args.pergunta,
        "resposta": resposta,
        "tempo_total_segundos": round(fim_total - inicio_total, 4),
        "tempo_carregamento_segundos": round(fim_carregamento - inicio_carregamento, 4),
        "tempo_geracao_segundos": round(fim_geracao - inicio_geracao, 4),
        "nfe": nfe,
        "parametros": {
            "mode": args.mode,
            "max_new_tokens": args.max_new_tokens,
            "block_length": args.block_length,
            "threshold": args.threshold,
            "temperature": args.temperature,
            "device": str(torch_device),
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
