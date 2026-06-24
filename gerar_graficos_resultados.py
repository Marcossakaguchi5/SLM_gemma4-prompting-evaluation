import argparse
import csv
from datetime import datetime
from pathlib import Path


OUTPUT_ROOT = Path("graficos_resultados")


def preparar_diretorio_saida(output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"rodada_{timestamp}"
    sufixo = 1
    while run_dir.exists():
        run_dir = output_root / f"rodada_{timestamp}_{sufixo:02d}"
        sufixo += 1
    run_dir.mkdir()
    return run_dir


def carregar_csv(path):
    with path.open("r", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def to_float(valor):
    try:
        if valor == "":
            return None
        return float(valor)
    except (TypeError, ValueError):
        return None


def rotulo_condicao(linha):
    """Distingue condicoes com a mesma abordagem em modelos diferentes."""
    condicao = str(linha.get("condicao", "")).strip()
    if condicao:
        return condicao
    modelo = str(linha.get("modelo", "")).strip()
    abordagem = str(linha.get("abordagem", "")).strip()
    return f"{modelo} / {abordagem}" if modelo else abordagem


def achar_metricas_dir(caminho):
    caminho = Path(caminho)
    if caminho.is_file():
        return caminho.parent
    return caminho


def grafico_barras_simples(plt, linhas, coluna_valor, titulo, ylabel, output):
    dados = [
        (rotulo_condicao(linha), to_float(linha.get(coluna_valor)))
        for linha in linhas
        if to_float(linha.get(coluna_valor)) is not None
    ]
    if not dados:
        return False

    labels = [item[0] for item in dados]
    valores = [item[1] for item in dados]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, valores, color="#3b82f6")
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Abordagem")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def grafico_barras_agrupadas(plt, linhas, coluna_valor, titulo, ylabel, output):
    linhas = [linha for linha in linhas if to_float(linha.get(coluna_valor)) is not None]
    if not linhas:
        return False

    datasets = sorted({linha["dataset"] for linha in linhas})
    condicoes = sorted({rotulo_condicao(linha) for linha in linhas})
    valores = {
        (linha["dataset"], rotulo_condicao(linha)): to_float(linha.get(coluna_valor))
        for linha in linhas
    }

    largura = 0.8 / max(len(condicoes), 1)
    x_base = list(range(len(datasets)))
    fig, ax = plt.subplots(figsize=(12, 6))

    for indice, condicao in enumerate(condicoes):
        xs = [x + (indice - (len(condicoes) - 1) / 2) * largura for x in x_base]
        ys = [
            valores.get((dataset, condicao), float("nan"))
            for dataset in datasets
        ]
        ax.bar(xs, ys, width=largura, label=condicao)

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    ax.set_xticks(x_base)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.legend(title="Modelo / condicao", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def grafico_custo_vs_acuracia(plt, linhas, output):
    pontos = []
    for linha in linhas:
        acuracia = to_float(linha.get("acuracia"))
        duracao = to_float(linha.get("duracao_media_segundos"))
        if acuracia is None or duracao is None:
            continue
        pontos.append((linha.get("dataset", ""), rotulo_condicao(linha), duracao, acuracia))

    if not pontos:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset, abordagem, duracao, acuracia in pontos:
        ax.scatter(duracao, acuracia, s=70)
        ax.annotate(f"{dataset}/{abordagem}", (duracao, acuracia), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title("Custo temporal vs. acuracia")
    ax.set_xlabel("Duracao media por resposta (s)")
    ax.set_ylabel("Acuracia")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def grafico_win_tie_loss(plt, linhas, output):
    if not linhas:
        return False
    labels = [
        f"{linha.get('modelo', '')}/{linha.get('dataset', '')}/{linha.get('abordagem', '')}"
        for linha in linhas
    ]
    wins = [to_float(linha.get("win_rate")) or 0.0 for linha in linhas]
    ties = [to_float(linha.get("tie_rate")) or 0.0 for linha in linhas]
    losses = [to_float(linha.get("loss_rate")) or 0.0 for linha in linhas]
    fig, ax = plt.subplots(figsize=(12, max(5, len(labels) * 0.35)))
    ys = list(range(len(labels)))
    ax.barh(ys, wins, label="Win", color="#16a34a")
    ax.barh(ys, ties, left=wins, label="Tie", color="#94a3b8")
    ax.barh(
        ys,
        losses,
        left=[win + tie for win, tie in zip(wins, ties)],
        label="Loss",
        color="#dc2626",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proporcao")
    ax.set_title("Win/Tie/Loss contra baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return True


def svg_escape(texto):
    return (
        str(texto)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def salvar_svg_barras_simples(linhas, coluna_valor, titulo, ylabel, output):
    dados = [
        (rotulo_condicao(linha), to_float(linha.get(coluna_valor)))
        for linha in linhas
        if to_float(linha.get(coluna_valor)) is not None
    ]
    if not dados:
        return False

    largura, altura = 900, 520
    margem_esq, margem_dir, margem_top, margem_bottom = 90, 30, 70, 120
    area_w = largura - margem_esq - margem_dir
    area_h = altura - margem_top - margem_bottom
    max_val = max(valor for _, valor in dados) or 1.0
    bar_w = area_w / max(len(dados), 1) * 0.65
    step = area_w / max(len(dados), 1)

    elementos = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{largura/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{svg_escape(titulo)}</text>',
        f'<text x="22" y="{altura/2}" transform="rotate(-90 22 {altura/2})" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(ylabel)}</text>',
        f'<line x1="{margem_esq}" y1="{margem_top + area_h}" x2="{margem_esq + area_w}" y2="{margem_top + area_h}" stroke="#111827"/>',
        f'<line x1="{margem_esq}" y1="{margem_top}" x2="{margem_esq}" y2="{margem_top + area_h}" stroke="#111827"/>',
    ]

    for indice, (label, valor) in enumerate(dados):
        x = margem_esq + indice * step + (step - bar_w) / 2
        h = (valor / max_val) * area_h
        y = margem_top + area_h - h
        elementos.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#3b82f6"/>')
        elementos.append(f'<text x="{x + bar_w/2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{valor:.3g}</text>')
        elementos.append(f'<text x="{x + bar_w/2:.2f}" y="{margem_top + area_h + 22}" transform="rotate(25 {x + bar_w/2:.2f} {margem_top + area_h + 22})" text-anchor="start" font-family="Arial" font-size="12">{svg_escape(label)}</text>')

    elementos.append("</svg>")
    output.write_text("\n".join(elementos), encoding="utf-8")
    return True


def salvar_svg_barras_agrupadas(linhas, coluna_valor, titulo, ylabel, output):
    linhas = [linha for linha in linhas if to_float(linha.get(coluna_valor)) is not None]
    if not linhas:
        return False

    datasets = sorted({linha["dataset"] for linha in linhas})
    condicoes = sorted({rotulo_condicao(linha) for linha in linhas})
    valores = {
        (linha["dataset"], rotulo_condicao(linha)): to_float(linha.get(coluna_valor))
        for linha in linhas
    }
    cores = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f59e0b", "#0891b2", "#4b5563"]
    largura, altura = 1100, 600
    margem_esq, margem_dir, margem_top, margem_bottom = 90, 170, 70, 140
    area_w = largura - margem_esq - margem_dir
    area_h = altura - margem_top - margem_bottom
    max_val = max(abs(valor) for valor in valores.values()) or 1.0
    min_val = min(0.0, min(valores.values()))
    max_axis = max(max_val, 0.0)
    span = max_axis - min_val or 1.0
    group_w = area_w / max(len(datasets), 1)
    bar_w = group_w / max(len(condicoes), 1) * 0.72

    def y_para_valor(valor):
        return margem_top + (max_axis - valor) / span * area_h

    y_zero = y_para_valor(0.0)
    elementos = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{largura/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{svg_escape(titulo)}</text>',
        f'<text x="22" y="{altura/2}" transform="rotate(-90 22 {altura/2})" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(ylabel)}</text>',
        f'<line x1="{margem_esq}" y1="{y_zero:.2f}" x2="{margem_esq + area_w}" y2="{y_zero:.2f}" stroke="#111827"/>',
        f'<line x1="{margem_esq}" y1="{margem_top}" x2="{margem_esq}" y2="{margem_top + area_h}" stroke="#111827"/>',
    ]

    for i, dataset in enumerate(datasets):
        group_x = margem_esq + i * group_w
        for j, condicao in enumerate(condicoes):
            valor = valores.get((dataset, condicao))
            if valor is None:
                continue
            x = group_x + j * (group_w / max(len(condicoes), 1)) + (group_w / max(len(condicoes), 1) - bar_w) / 2
            y_val = y_para_valor(valor)
            y = min(y_val, y_zero)
            h = abs(y_zero - y_val)
            cor = cores[j % len(cores)]
            elementos.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{cor}"/>')
        elementos.append(f'<text x="{group_x + group_w/2:.2f}" y="{margem_top + area_h + 24}" transform="rotate(20 {group_x + group_w/2:.2f} {margem_top + area_h + 24})" text-anchor="start" font-family="Arial" font-size="12">{svg_escape(dataset)}</text>')

    for j, condicao in enumerate(condicoes):
        y = margem_top + j * 22
        cor = cores[j % len(cores)]
        elementos.append(f'<rect x="{largura - margem_dir + 20}" y="{y}" width="14" height="14" fill="{cor}"/>')
        elementos.append(f'<text x="{largura - margem_dir + 42}" y="{y + 12}" font-family="Arial" font-size="12">{svg_escape(condicao)}</text>')

    elementos.append("</svg>")
    output.write_text("\n".join(elementos), encoding="utf-8")
    return True


def salvar_svg_custo_vs_acuracia(linhas, output):
    pontos = []
    for linha in linhas:
        acuracia = to_float(linha.get("acuracia"))
        duracao = to_float(linha.get("duracao_media_segundos"))
        if acuracia is None or duracao is None:
            continue
        pontos.append((linha.get("dataset", ""), rotulo_condicao(linha), duracao, acuracia))
    if not pontos:
        return False

    largura, altura = 1000, 600
    margem_esq, margem_dir, margem_top, margem_bottom = 90, 40, 70, 80
    area_w = largura - margem_esq - margem_dir
    area_h = altura - margem_top - margem_bottom
    max_x = max(p[2] for p in pontos) or 1.0
    min_x = min(p[2] for p in pontos)
    max_y = max(p[3] for p in pontos) or 1.0
    min_y = min(p[3] for p in pontos)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0

    elementos = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{largura/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Custo temporal vs. acuracia</text>',
        f'<line x1="{margem_esq}" y1="{margem_top + area_h}" x2="{margem_esq + area_w}" y2="{margem_top + area_h}" stroke="#111827"/>',
        f'<line x1="{margem_esq}" y1="{margem_top}" x2="{margem_esq}" y2="{margem_top + area_h}" stroke="#111827"/>',
        f'<text x="{largura/2}" y="{altura - 18}" text-anchor="middle" font-family="Arial" font-size="14">Duracao media por resposta (s)</text>',
        f'<text x="22" y="{altura/2}" transform="rotate(-90 22 {altura/2})" text-anchor="middle" font-family="Arial" font-size="14">Acuracia</text>',
    ]

    for dataset, abordagem, duracao, acuracia in pontos:
        x = margem_esq + (duracao - min_x) / span_x * area_w
        y = margem_top + (max_y - acuracia) / span_y * area_h
        label = f"{dataset}/{abordagem}"
        elementos.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#3b82f6"/>')
        elementos.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-family="Arial" font-size="11">{svg_escape(label)}</text>')

    elementos.append("</svg>")
    output.write_text("\n".join(elementos), encoding="utf-8")
    return True


def salvar_svg_win_tie_loss(linhas, output):
    if not linhas:
        return False

    labels = [
        f"{linha.get('modelo', '')}/{linha.get('dataset', '')}/{linha.get('abordagem', '')}"
        for linha in linhas
    ]
    largura = 1100
    altura = max(420, 120 + len(linhas) * 34)
    margem_esq, margem_dir, margem_top, margem_bottom = 250, 50, 70, 70
    area_w = largura - margem_esq - margem_dir
    area_h = altura - margem_top - margem_bottom
    barra_h = min(22, area_h / max(len(linhas), 1) * 0.7)
    passo = area_h / max(len(linhas), 1)
    cores = {
        "win": "#16a34a",
        "tie": "#94a3b8",
        "loss": "#dc2626",
    }
    elementos = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" viewBox="0 0 {largura} {altura}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{largura/2}" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">Win/Tie/Loss contra baseline</text>',
    ]

    for indice, (linha, label) in enumerate(zip(linhas, labels)):
        y = margem_top + indice * passo + (passo - barra_h) / 2
        win = to_float(linha.get("win_rate")) or 0.0
        tie = to_float(linha.get("tie_rate")) or 0.0
        loss = to_float(linha.get("loss_rate")) or 0.0
        x = margem_esq
        for nome, valor in (("win", win), ("tie", tie), ("loss", loss)):
            segmento = area_w * valor
            elementos.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{segmento:.2f}" '
                f'height="{barra_h:.2f}" fill="{cores[nome]}"/>'
            )
            x += segmento
        elementos.append(
            f'<text x="{margem_esq - 10}" y="{y + barra_h * 0.75:.2f}" '
            f'text-anchor="end" font-family="Arial" font-size="12">{svg_escape(label)}</text>'
        )

    legenda_x = margem_esq
    for nome, rotulo in (("win", "Win"), ("tie", "Tie"), ("loss", "Loss")):
        elementos.append(
            f'<rect x="{legenda_x}" y="{altura - 38}" width="14" height="14" '
            f'fill="{cores[nome]}"/>'
        )
        elementos.append(
            f'<text x="{legenda_x + 20}" y="{altura - 26}" '
            f'font-family="Arial" font-size="12">{rotulo}</text>'
        )
        legenda_x += 90

    elementos.append("</svg>")
    output.write_text("\n".join(elementos), encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gera graficos a partir das metricas processadas.")
    parser.add_argument("metricas", help="Diretorio gerado por processar_resultados.py ou um CSV de metricas.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    metricas_dir = achar_metricas_dir(args.metricas)
    por_abordagem_path = metricas_dir / "metricas_por_abordagem.csv"
    por_dataset_path = metricas_dir / "metricas_por_dataset_abordagem.csv"
    win_tie_loss_path = metricas_dir / "win_tie_loss_vs_baseline.csv"
    position_bias_path = metricas_dir / "auditoria_position_bias.csv"

    if not por_abordagem_path.exists() or not por_dataset_path.exists():
        raise FileNotFoundError(
            "Nao encontrei metricas_por_abordagem.csv e metricas_por_dataset_abordagem.csv."
        )

    por_abordagem = carregar_csv(por_abordagem_path)
    por_dataset = carregar_csv(por_dataset_path)
    win_tie_loss = carregar_csv(win_tie_loss_path) if win_tie_loss_path.exists() else []
    position_bias = carregar_csv(position_bias_path) if position_bias_path.exists() else []
    run_dir = preparar_diretorio_saida(Path(args.output_root))

    gerados = []
    if plt is not None:
        if grafico_barras_simples(
            plt,
            por_abordagem,
            "acuracia",
            "Acuracia por abordagem",
            "Acuracia",
            run_dir / "acuracia_por_abordagem.png",
        ):
            gerados.append("acuracia_por_abordagem.png")

        if grafico_barras_simples(
            plt,
            por_abordagem,
            "duracao_media_segundos",
            "Tempo medio por abordagem",
            "Segundos",
            run_dir / "tempo_medio_por_abordagem.png",
        ):
            gerados.append("tempo_medio_por_abordagem.png")

        if grafico_barras_agrupadas(
            plt,
            por_dataset,
            "acuracia",
            "Acuracia por dataset e abordagem",
            "Acuracia",
            run_dir / "acuracia_por_dataset_abordagem.png",
        ):
            gerados.append("acuracia_por_dataset_abordagem.png")

        if grafico_barras_agrupadas(
            plt,
            por_dataset,
            "delta_acuracia_vs_base",
            "Delta de acuracia contra base",
            "Delta",
            run_dir / "delta_vs_base.png",
        ):
            gerados.append("delta_vs_base.png")

        if grafico_custo_vs_acuracia(plt, por_dataset, run_dir / "custo_vs_acuracia.png"):
            gerados.append("custo_vs_acuracia.png")

        graficos_agrupados = [
            (
                "answer_match_rate",
                "Answer Match por dataset e abordagem",
                "Answer Match",
                "answer_match.png",
            ),
            (
                "exact_match_rate",
                "Exact Match por dataset e abordagem",
                "Exact Match",
                "exact_match.png",
            ),
            (
                "symbolic_equivalence_rate",
                "Equivalencia simbolica em matematica",
                "Symbolic Equivalence",
                "equivalencia_simbolica.png",
            ),
            (
                "truthfulness_rate",
                "Truthfulness por abordagem",
                "Truthfulness",
                "truthfulness.png",
            ),
            (
                "informativeness_rate",
                "Informativeness por abordagem",
                "Informativeness",
                "informativeness.png",
            ),
            (
                "oracle_at_3_rate",
                "Oracle@3 da estrategia multi-trajetoria",
                "Oracle@3",
                "oracle_at_3.png",
            ),
            (
                "accuracy_per_second",
                "Accuracy per Second",
                "Accuracy/s",
                "accuracy_per_second.png",
            ),
            (
                "gain_per_extra_call",
                "Gain per Extra Call",
                "Ganho/chamada",
                "gain_per_extra_call.png",
            ),
        ]
        for coluna, titulo, ylabel, nome in graficos_agrupados:
            if grafico_barras_agrupadas(
                plt,
                por_dataset,
                coluna,
                titulo,
                ylabel,
                run_dir / nome,
            ):
                gerados.append(nome)

        if position_bias and grafico_barras_agrupadas(
            plt,
            position_bias,
            "position_bias_rate",
            "Position Bias Rate do juiz",
            "Taxa de divergencia",
            run_dir / "position_bias_rate.png",
        ):
            gerados.append("position_bias_rate.png")

        if grafico_win_tie_loss(
            plt,
            win_tie_loss,
            run_dir / "win_tie_loss_vs_baseline.png",
        ):
            gerados.append("win_tie_loss_vs_baseline.png")
    else:
        if salvar_svg_barras_simples(
            por_abordagem,
            "acuracia",
            "Acuracia por abordagem",
            "Acuracia",
            run_dir / "acuracia_por_abordagem.svg",
        ):
            gerados.append("acuracia_por_abordagem.svg")

        if salvar_svg_barras_simples(
            por_abordagem,
            "duracao_media_segundos",
            "Tempo medio por abordagem",
            "Segundos",
            run_dir / "tempo_medio_por_abordagem.svg",
        ):
            gerados.append("tempo_medio_por_abordagem.svg")

        if salvar_svg_barras_agrupadas(
            por_dataset,
            "acuracia",
            "Acuracia por dataset e abordagem",
            "Acuracia",
            run_dir / "acuracia_por_dataset_abordagem.svg",
        ):
            gerados.append("acuracia_por_dataset_abordagem.svg")

        if salvar_svg_barras_agrupadas(
            por_dataset,
            "delta_acuracia_vs_base",
            "Delta de acuracia contra base",
            "Delta",
            run_dir / "delta_vs_base.svg",
        ):
            gerados.append("delta_vs_base.svg")

        if salvar_svg_custo_vs_acuracia(por_dataset, run_dir / "custo_vs_acuracia.svg"):
            gerados.append("custo_vs_acuracia.svg")

        graficos_agrupados = [
            (
                "answer_match_rate",
                "Answer Match",
                "Answer Match",
                "answer_match.svg",
            ),
            ("exact_match_rate", "Exact Match", "Exact Match", "exact_match.svg"),
            (
                "symbolic_equivalence_rate",
                "Equivalencia simbolica",
                "Symbolic Equivalence",
                "equivalencia_simbolica.svg",
            ),
            (
                "truthfulness_rate",
                "Truthfulness",
                "Truthfulness",
                "truthfulness.svg",
            ),
            (
                "informativeness_rate",
                "Informativeness",
                "Informativeness",
                "informativeness.svg",
            ),
            ("oracle_at_3_rate", "Oracle@3", "Oracle@3", "oracle_at_3.svg"),
            (
                "accuracy_per_second",
                "Accuracy per Second",
                "Accuracy/s",
                "accuracy_per_second.svg",
            ),
            (
                "gain_per_extra_call",
                "Gain per Extra Call",
                "Ganho/chamada",
                "gain_per_extra_call.svg",
            ),
        ]
        for coluna, titulo, ylabel, nome in graficos_agrupados:
            if salvar_svg_barras_agrupadas(
                por_dataset,
                coluna,
                titulo,
                ylabel,
                run_dir / nome,
            ):
                gerados.append(nome)

        if position_bias and salvar_svg_barras_agrupadas(
            position_bias,
            "position_bias_rate",
            "Position Bias Rate do juiz",
            "Taxa de divergencia",
            run_dir / "position_bias_rate.svg",
        ):
            gerados.append("position_bias_rate.svg")

        if salvar_svg_win_tie_loss(
            win_tie_loss,
            run_dir / "win_tie_loss_vs_baseline.svg",
        ):
            gerados.append("win_tie_loss_vs_baseline.svg")

    print(f"Graficos salvos em: {run_dir}")
    for nome in gerados:
        print(f"- {nome}")


if __name__ == "__main__":
    main()
