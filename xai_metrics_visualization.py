"""
xai_metrics_visualization.py
=============================
Geração de visualizações e testes estatísticos sobre as métricas espaciais
calculadas por xai_quantitative_evaluation.py (Sec. 4.2.3 do TCC).

Produz:
  - Boxplots das métricas espaciais por modelo e técnica CAM
  - Teste de Wilcoxon pareado entre configurações (mesmas 89 imagens)
  - Correção de múltiplas comparações por Holm-Bonferroni (α = 0,05)
  - δ de Cliff como medida de tamanho de efeito
  - CSVs com resultados estatísticos completos

Saídas em xai_visual_analysis/:
  boxplot_{metric}_gradcam.png
  boxplot_{metric}_scorecam.png
  statistical_tests.csv
  effect_sizes.csv

Uso:
  python xai_metrics_visualization.py
         [--input xai_quantitative_results.csv]
         [--output xai_visual_analysis]

Referência:
  Miranda, G.P. (2026). TCC — IFES Campus Serra.
"""

import argparse
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Estilo global ────────────────────────────────────────────────────────────

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size":    11,
    "figure.dpi":   300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

METRICS  = ["iou", "precision", "recall", "f1"]
CAM_TYPES = ["gradcam", "scorecam"]
ALPHA    = 0.05    # nível de significância (Sec. 4.2.3)


# ─── Carregamento ─────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carrega o CSV gerado por xai_quantitative_evaluation.py.
    Normaliza nome da coluna f1_score → f1 se necessário.
    """
    df = pd.read_csv(csv_path)
    if "f1_score" in df.columns:
        df = df.rename(columns={"f1_score": "f1"})
    missing = [m for m in METRICS if m not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    print(f"  {len(df)} registros carregados "
          f"({df['image'].nunique()} imagens, "
          f"{df['model'].nunique()} modelos)")
    return df


# ─── δ de Cliff (vetorizado) ──────────────────────────────────────────────────

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula o δ de Cliff entre dois vetores independentes.
    Implementação vetorizada: O(n·m) sem loops Python.

    Escala convencional:
      |δ| < 0.15  — desprezível
      0.15–0.33   — pequeno
      0.33–0.47   — moderado
      > 0.47      — grande
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # produto externo: x[i] vs y[j] para todos os pares
    dominance = np.sign(x[:, None] - y[None, :])
    return float(dominance.mean())


def effect_label(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.15:
        return "desprezível"
    elif ad < 0.33:
        return "pequeno"
    elif ad < 0.47:
        return "moderado"
    return "grande"


# ─── Holm-Bonferroni ──────────────────────────────────────────────────────────

def holm_bonferroni(p_values: list[float],
                    alpha: float = ALPHA) -> list[bool]:
    """
    Correção de Holm-Bonferroni para comparações múltiplas (Sec. 4.2.3).
    Retorna lista booleana: True = hipótese nula rejeitada (significativo).
    """
    n = len(p_values)
    if n == 0:
        return []
    order   = np.argsort(p_values)
    ranked  = np.array(p_values)[order]
    reject  = np.zeros(n, dtype=bool)
    for k, idx in enumerate(order):
        threshold = alpha / (n - k)
        if ranked[k] <= threshold:
            reject[idx] = True
        else:
            # Uma vez que falha, todos os subsequentes também falham
            break
    return reject.tolist()


# ─── Testes de Wilcoxon pareados ──────────────────────────────────────────────

def run_wilcoxon_tests(df: pd.DataFrame,
                       cam_type: str,
                       metric: str) -> pd.DataFrame:
    """
    Aplica o teste de Wilcoxon de postos sinalizados (pareado) entre todos
    os pares de modelos para um dado (cam_type, metric).

    O teste é pareado porque as mesmas 89 imagens do DIARETDB1 foram
    processadas por todos os modelos (Sec. 4.2.3).

    Aplica correção de Holm-Bonferroni sobre os p-valores resultantes.

    Retorna DataFrame com colunas:
      cam, metric, model_1, model_2, n_pairs,
      statistic, p_value, p_corrected, significant, cliffs_delta, effect
    """
    sub     = df[df["cam"] == cam_type].copy()
    models  = sorted(sub["model"].unique())
    pairs   = list(itertools.combinations(models, 2))

    rows = []
    for m1, m2 in pairs:
        # Alinha pelos mesmos image_id (garantindo pareamento)
        d1 = sub[sub["model"] == m1][["image", metric]].set_index("image")
        d2 = sub[sub["model"] == m2][["image", metric]].set_index("image")
        common = d1.index.intersection(d2.index)

        if len(common) < 10:
            continue  # insuficiente para o teste

        x = d1.loc[common, metric].values
        y = d2.loc[common, metric].values

        diff = x - y
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")

        cd = cliffs_delta(x, y)

        rows.append({
            "cam":          cam_type,
            "metric":       metric,
            "model_1":      m1,
            "model_2":      m2,
            "n_pairs":      len(common),
            "statistic":    round(stat, 4),
            "p_value":      round(p, 6),
            "cliffs_delta": round(cd, 4),
            "effect":       effect_label(cd),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Holm-Bonferroni sobre todos os p-valores do grupo
    reject = holm_bonferroni(result["p_value"].tolist())
    result["significant"] = reject

    return result


# ─── Boxplots ─────────────────────────────────────────────────────────────────

def publication_boxplot(df: pd.DataFrame,
                        metric: str,
                        cam_type: str,
                        output_dir: str) -> None:
    """
    Gera boxplot de uma métrica espacial por modelo, para uma técnica CAM.

    Eixo x: modelo (rótulo abreviado)
    Eixo y: valor da métrica
    """
    sub = df[df["cam"] == cam_type].copy()

    # Abreviação dos modelos para melhor legibilidade no eixo x
    sub["model_short"] = (
        sub["model"]
        .str.replace("diagnostico_",    "Dx·",   regex=False)
        .str.replace("classificacao_",  "Cls·",  regex=False)
        .str.replace("densenet121",     "DN121", regex=False)
        .str.replace("efficientnetb3",  "EffB3", regex=False)
        .str.replace("resnet50",        "RN50",  regex=False)
        .str.replace("vgg16",           "VGG16", regex=False)
        .str.replace("_ben_graham",     "+BG",   regex=False)
        .str.replace("_clahe",          "+CLAHE",regex=False)
        .str.replace("_original",       "+Orig", regex=False)
    )

    order = sorted(sub["model_short"].unique())

    fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.6), 5))

    sns.boxplot(
        data=sub,
        x="model_short",
        y=metric,
        order=order,
        width=0.6,
        fliersize=2,
        linewidth=1.2,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"{metric.upper()} — {cam_type.capitalize()} (DIARETDB1, n=89)",
        fontsize=12, fontweight="bold"
    )
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"boxplot_{metric}_{cam_type}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def summary_heatmap(summary: pd.DataFrame,
                    metric: str,
                    cam_type: str,
                    output_dir: str) -> None:
    """
    Heatmap modelo × métrica mostrando as médias (para o cam_type dado).
    Útil para inspecionar visualmente padrões de desempenho relativo.
    """
    sub = summary[summary["cam"] == cam_type][["model", metric]].copy()
    sub = sub.set_index("model")

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(4, max(4, len(sub) * 0.4)))
    sns.heatmap(
        sub,
        annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0, vmax=sub[metric].max(),
        ax=ax, cbar_kws={"shrink": 0.7},
        annot_kws={"size": 9},
    )
    ax.set_title(
        f"Média {metric.upper()} — {cam_type.capitalize()}",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("")
    plt.tight_layout()
    fname = os.path.join(output_dir, f"heatmap_{metric}_{cam_type}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main_pipeline(input_csv: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nxai_retinopathy — Visualização e Testes Estatísticos")
    print(f"Input : {input_csv}")
    print(f"Output: {output_dir}\n")

    # ── Carrega dados ──────────────────────────────────────────────────────
    df = load_data(input_csv)

    # ── Resumo por (modelo, cam) ───────────────────────────────────────────
    summary = (
        df.groupby(["model", "cam"])[METRICS]
        .mean()
        .round(4)
        .reset_index()
    )
    summary_path = os.path.join(output_dir, "summary_means.csv")
    summary.to_csv(summary_path, index=False)
    print("Resumo de médias:\n")
    print(summary.to_string(index=False))

    # ── Boxplots ───────────────────────────────────────────────────────────
    print("\nGerando boxplots...")
    for cam in CAM_TYPES:
        for metric in METRICS:
            if df[df["cam"] == cam].empty:
                continue
            publication_boxplot(df, metric, cam, output_dir)
            summary_heatmap(summary, metric, cam, output_dir)
    print(f"  Boxplots e heatmaps salvos em: {output_dir}")

    # ── Testes de Wilcoxon + Holm-Bonferroni ──────────────────────────────
    print("\nExecutando testes de Wilcoxon (pareados, Holm-Bonferroni)...")
    stats_frames = []
    for cam in CAM_TYPES:
        for metric in METRICS:
            result = run_wilcoxon_tests(df, cam, metric)
            if not result.empty:
                stats_frames.append(result)

    if stats_frames:
        stats_df = pd.concat(stats_frames, ignore_index=True)
        stats_path = os.path.join(output_dir, "statistical_tests.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"  Testes salvos em: {stats_path}")

        # Resumo de comparações significativas
        sig = stats_df[stats_df["significant"]]
        print(f"\n  Comparações significativas (α={ALPHA}, Holm-Bonferroni): "
              f"{len(sig)} / {len(stats_df)}")
        if not sig.empty:
            print(sig[["cam", "metric", "model_1", "model_2",
                        "p_value", "cliffs_delta", "effect"]].to_string(index=False))
    else:
        print("  [AVISO] Nenhum resultado estatístico gerado.")

    print("\n✔ Análise concluída.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualização e testes estatísticos XAI — xai_retinopathy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default="xai_quantitative_results.csv",
        help="CSV gerado por xai_quantitative_evaluation.py",
    )
    parser.add_argument(
        "--output", type=str, default="xai_visual_analysis",
        help="Diretório de saída para figuras e CSVs estatísticos",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"CSV não encontrado: {args.input}")
    main_pipeline(args.input, args.output)