"""
xai_final_ranking.py
====================
Integração dos três eixos de avaliação e geração do ranking multicritério
(Sec. 3.7.4 do TCC).

Fórmula documentada (pesos uniformes):
    Score_final = (1/3) * S_perf + (1/3) * S_estab + (1/3) * S_anat

Onde:
    S_perf  — AUC-ROC do modelo no conjunto de teste, normalizado min-max
    S_estab — SSIM médio entre Grad-CAM e Score-CAM, normalizado min-max
    S_anat  — IoU médio do Score-CAM vs máscaras GT DIARETDB1, normalizado min-max

Entradas requeridas:
    --training_csv   outputs/results/training_results.csv   (training_hub.py)
    --similarity_csv xai_similarity_results_summary.csv     (xai_similarity_analysis.py)
    --xai_csv        xai_quantitative_results.csv           (xai_quantitative_evaluation.py)

Saídas em xai_final_analysis/:
    ranking_diagnostico.csv
    ranking_classificacao.csv
    radar_contraste_diag.png    ← 4 perfis contrastantes — diagnóstico
    radar_contraste_class.png   ← 4 perfis contrastantes — classificação

Uso:
    python xai_final_ranking.py
           --training_csv outputs/results/training_results.csv
           --similarity_csv xai_similarity_results_summary.csv
           --xai_csv xai_quantitative_results.csv
           [--output xai_final_analysis]

Referência:
    Miranda, G.P. (2026). TCC — IFES Campus Serra.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Constantes documentadas ──────────────────────────────────────────────────

WEIGHT = 1.0 / 3.0    # pesos uniformes (Sec. 3.7.4, Eq. score_final)

RADAR_CATEGORIES = [
    r"$S_{\mathrm{perf}}$" + "\n(AUC norm.)",
    r"$S_{\mathrm{estab}}$" + "\n(SSIM norm.)",
    r"$S_{\mathrm{anat}}$" + "\n(IoU norm.)",
]
SCORE_COLS = ["s_perf", "s_estab", "s_anat"]

# 4 configurações contrastantes para o radar (documentadas no TCC)
CONTRAST_MODELS = [
    "resnet50_ben_graham",
    "efficientnetb3_clahe",
    "densenet121_clahe",
    "vgg16_clahe",
]
CONTRAST_COLORS = ["#1565C0", "#F57F17", "#2E7D32", "#B71C1C"]


# ─── Parsing de nome de modelo ────────────────────────────────────────────────

def split_model_name(model_full: str) -> tuple[str, str, str]:
    """
    Divide o nome completo do modelo em (task, architecture, preprocessing).

    Exemplos:
      'diagnostico_resnet50_ben_graham'  → ('diagnostico', 'resnet50', 'ben_graham')
      'classificacao_vgg16_clahe'        → ('classificacao', 'vgg16', 'clahe')

    Lida corretamente com pré-processamentos de múltiplos tokens (ben_graham).
    """
    # Tarefas e arquiteturas conhecidas (sem ambiguidade)
    tasks  = {"diagnostico", "classificacao"}
    archs  = {"densenet121", "efficientnetb3", "resnet50", "vgg16"}
    preps  = {"original", "clahe", "ben_graham"}

    parts = model_full.split("_")
    task = arch = prep = ""

    # Identifica task (sempre primeiro token)
    if parts[0] in tasks:
        task = parts[0]
        rest = parts[1:]
    else:
        rest = parts

    # Identifica arquitetura (pode ter 1 ou 2 tokens: densenet121, efficientnetb3)
    for length in (2, 1):
        candidate = "_".join(rest[:length])
        if candidate in archs:
            arch = candidate
            rest = rest[length:]
            break

    # Pré-processamento = resto
    prep = "_".join(rest)
    if prep not in preps:
        prep = rest[0] if rest else ""

    return task, arch, prep


# ─── Carregamento e integração ────────────────────────────────────────────────

def load_and_merge(training_csv: str,
                   similarity_csv: str,
                   xai_csv: str) -> pd.DataFrame:
    """
    Integra os três CSVs de entrada, alinhando por (task, model_key).

    model_key = '{architecture}_{preprocessing}'  (sem prefixo de task)

    Colunas resultantes por linha (um modelo por tarefa):
        task, model_key, model_full, architecture, preprocessing,
        auc_roc, ssim_mean, iou_scorecam_mean
    """
    # ── S_perf: AUC-ROC do training_hub ──────────────────────────────────
    df_train = pd.read_csv(training_csv)
    # Colunas esperadas: task, architecture, preprocessing, auc_roc
    if "auc_roc" not in df_train.columns:
        raise ValueError("training_results.csv deve conter coluna 'auc_roc'.")

    df_train["task"]      = df_train["task"].str.strip()
    df_train["model_key"] = (df_train["architecture"].str.strip() + "_"
                             + df_train["preprocessing"].str.strip())
    df_perf = df_train[["task", "model_key", "auc_roc"]].copy()

    # ── S_estab: SSIM médio do xai_similarity_analysis ───────────────────
    df_sim = pd.read_csv(similarity_csv)
    # Colunas esperadas: model, ssim
    if "ssim" not in df_sim.columns:
        raise ValueError("similarity CSV deve conter coluna 'ssim'.")

    # model pode ser 'diagnostico_resnet50_ben_graham' ou 'resnet50_ben_graham'
    # Normaliza para extrair task + model_key
    rows_sim = []
    for _, row in df_sim.iterrows():
        task, arch, prep = split_model_name(row["model"])
        rows_sim.append({
            "task":      task,
            "model_key": f"{arch}_{prep}",
            "ssim_mean": row["ssim"],
        })
    df_estab = pd.DataFrame(rows_sim)

    # ── S_anat: IoU Score-CAM do xai_quantitative_evaluation ─────────────
    df_xai = pd.read_csv(xai_csv)
    # Filtra apenas Score-CAM (S_anat usa Score-CAM, Sec. 3.7.4)
    df_score = df_xai[df_xai["cam"] == "scorecam"].copy()

    rows_anat = []
    for model_full, grp in df_score.groupby("model"):
        task, arch, prep = split_model_name(model_full)
        rows_anat.append({
            "task":             task,
            "model_key":        f"{arch}_{prep}",
            "iou_scorecam_mean": grp["iou"].mean(),
        })
    df_anat = pd.DataFrame(rows_anat)

    # ── Merge ─────────────────────────────────────────────────────────────
    df = df_perf.merge(df_estab, on=["task", "model_key"], how="outer")
    df = df.merge(df_anat,  on=["task", "model_key"], how="outer")

    # Metadados por modelo
    df["model_full"] = df["task"] + "_" + df["model_key"]
    df[["arch_col", "prep_col"]] = df["model_key"].str.split(
        "_", n=1, expand=True
    )
    df = df.rename(columns={"arch_col": "architecture",
                             "prep_col": "preprocessing"})

    missing = df[df[["auc_roc", "ssim_mean", "iou_scorecam_mean"]].isna().any(axis=1)]
    if not missing.empty:
        print(f"  [AVISO] {len(missing)} modelos com dados incompletos:")
        print(missing[["model_full"]].to_string(index=False))

    return df.dropna(subset=["auc_roc", "ssim_mean", "iou_scorecam_mean"])


# ─── Normalização min-max ─────────────────────────────────────────────────────

def minmax(series: pd.Series) -> pd.Series:
    """
    Normalização min-max independente por eixo (Sec. 3.7.4).
    O modelo com melhor valor recebe 1.000, o de pior valor 0.000.
    """
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-8:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


# ─── Ranking por tarefa ───────────────────────────────────────────────────────

def compute_ranking(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Calcula o Score_final = (1/3)*S_perf + (1/3)*S_estab + (1/3)*S_anat
    para todos os modelos de uma tarefa, ordenados de forma decrescente.
    """
    sub = df[df["task"] == task].copy()

    sub["s_perf"]  = minmax(sub["auc_roc"])
    sub["s_estab"] = minmax(sub["ssim_mean"])
    sub["s_anat"]  = minmax(sub["iou_scorecam_mean"])

    sub["score_final"] = WEIGHT * (sub["s_perf"] + sub["s_estab"] + sub["s_anat"])
    sub = sub.sort_values("score_final", ascending=False).reset_index(drop=True)
    sub["rank"] = sub.index + 1

    return sub


# ─── Radar de perfis contrastantes ───────────────────────────────────────────

def plot_radar_contrast(df_rank: pd.DataFrame,
                        task: str,
                        output_dir: str) -> None:
    """
    Gera radar_contraste_{task}.png com 4 perfis contrastantes
    nos eixos S_perf, S_estab, S_anat (Sec. 4.7 do TCC).

    Modelos selecionados: CONTRAST_MODELS (por model_key).
    Se algum não existir na tarefa, é ignorado silenciosamente.
    """
    n_axes  = len(SCORE_COLS)
    angles  = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    # Grid rings
    for r_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles_c, [r_val] * (n_axes + 1), color="#DDDDDD", lw=0.9, zorder=1)

    plotted = 0
    for model_key, color in zip(CONTRAST_MODELS, CONTRAST_COLORS):
        row = df_rank[df_rank["model_key"] == model_key]
        if row.empty:
            continue

        vals   = row[SCORE_COLS].values[0].tolist()
        vals_c = vals + [vals[0]]
        label  = model_key.replace("_", " + ").replace("ben + graham", "ben graham")

        ax.fill(angles_c, vals_c, alpha=0.12, color=color, zorder=2)
        ax.plot(angles_c, vals_c, color=color, lw=2.5, label=label, zorder=3)
        ax.scatter(angles, vals, color=color, s=48, zorder=4)
        plotted += 1

    ax.set_thetagrids(np.degrees(angles), RADAR_CATEGORIES, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0,2", "0,4", "0,6", "0,8", "1,0"],
                       fontsize=8.5, color="#666666")
    ax.grid(color="#DDDDDD", linestyle="--", lw=0.7)
    ax.spines["polar"].set_color("#CCCCCC")

    task_label = "Diagnóstico Binário" if task == "diagnostico" else "Classificação de Severidade"
    plt.title(
        f"Perfis Interpretativos — {task_label}\n(4 configurações selecionadas)",
        fontsize=12.5, fontweight="bold", pad=25, color="#1A1A2E",
    )
    plt.legend(
        loc="upper right", bbox_to_anchor=(1.45, 1.22),
        fontsize=9.5, framealpha=0.95, edgecolor="#CCCCCC",
    )
    plt.tight_layout()

    fname = os.path.join(output_dir, f"radar_contraste_{task}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✔ {os.path.basename(fname)} ({plotted} modelos plotados)")


# ─── Barplot horizontal do ranking ───────────────────────────────────────────

def plot_ranking_bar(df_rank: pd.DataFrame,
                     task: str,
                     output_dir: str) -> None:
    """
    Barplot horizontal com o Score_final de cada modelo,
    colorido por componente (S_perf, S_estab, S_anat empilhados).
    """
    df_plot = df_rank.sort_values("score_final").copy()
    labels  = (df_plot["architecture"] + " + " +
               df_plot["preprocessing"].str.replace("_", " "))

    fig, ax = plt.subplots(figsize=(9, max(5, len(df_plot) * 0.45)))

    bar_colors = {"s_perf": "#1565C0", "s_estab": "#2E7D32", "s_anat": "#B71C1C"}
    bottoms = np.zeros(len(df_plot))
    for col, color in bar_colors.items():
        vals = (df_plot[col] * WEIGHT).values
        ax.barh(labels, vals, left=bottoms, color=color,
                alpha=0.82, label=col.replace("_", " ").upper())
        bottoms += vals

    ax.set_xlabel("Score Final Multicritério", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_title(
        f"Ranking Multicritério — {'Diagnóstico Binário' if task == 'diagnostico' else 'Classificação de Severidade'}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    fname = os.path.join(output_dir, f"ranking_bar_{task}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✔ {os.path.basename(fname)}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ranking multicritério XAI — xai_retinopathy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--training_csv", type=str,
        default="outputs/results/training_results.csv",
        help="CSV de resultados do training_hub.py (contém auc_roc por modelo)",
    )
    parser.add_argument(
        "--similarity_csv", type=str,
        default="xai_similarity_results_summary.csv",
        help="CSV de resumo do xai_similarity_analysis.py (contém ssim por modelo)",
    )
    parser.add_argument(
        "--xai_csv", type=str,
        default="xai_quantitative_results.csv",
        help="CSV do xai_quantitative_evaluation.py (contém IoU por imagem e modelo)",
    )
    parser.add_argument(
        "--output", type=str, default="xai_final_analysis",
        help="Diretório de saída para rankings e radares",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    for path, label in [
        (args.training_csv,  "training_csv"),
        (args.similarity_csv,"similarity_csv"),
        (args.xai_csv,       "xai_csv"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Arquivo '{label}' não encontrado: {path}")

    os.makedirs(args.output, exist_ok=True)

    print("\nxai_retinopathy — Ranking Multicritério")
    print(f"  S_perf  ← {args.training_csv}")
    print(f"  S_estab ← {args.similarity_csv}")
    print(f"  S_anat  ← {args.xai_csv}  (Score-CAM only)")
    print(f"  Output  → {args.output}\n")

    # ── Integração ────────────────────────────────────────────────────────
    df = load_and_merge(args.training_csv, args.similarity_csv, args.xai_csv)
    print(f"Modelos integrados: {len(df)} "
          f"({df[df['task']=='diagnostico'].shape[0]} diag + "
          f"{df[df['task']=='classificacao'].shape[0]} class)\n")

    # ── Ranking por tarefa ────────────────────────────────────────────────
    for task in ["diagnostico", "classificacao"]:
        print(f"{'='*60}")
        print(f" Tarefa: {task}")
        print(f"{'='*60}")

        df_rank = compute_ranking(df, task)

        # Exporta CSV
        csv_path = os.path.join(args.output, f"ranking_{task}.csv")
        df_rank[[
            "rank", "model_full", "architecture", "preprocessing",
            "auc_roc", "ssim_mean", "iou_scorecam_mean",
            "s_perf", "s_estab", "s_anat", "score_final",
        ]].to_csv(csv_path, index=False)
        print(f"  ✔ {os.path.basename(csv_path)}")

        # Exibe ranking no terminal
        display_cols = ["rank", "model_key", "s_perf", "s_estab", "s_anat", "score_final"]
        available    = [c for c in display_cols if c in df_rank.columns]
        print(df_rank[available].to_string(index=False))
        print()

        # Gráficos
        plot_ranking_bar(df_rank, task, args.output)
        plot_radar_contrast(df_rank, task, args.output)

    print("✔ Ranking multicritério concluído.")


if __name__ == "__main__":
    main()