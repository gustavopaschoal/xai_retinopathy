"""
xai_similarity_analysis.py
===========================
Cálculo da estabilidade interpretativa entre os mapas Grad-CAM e Score-CAM
gerados para cada modelo (Sec. 3.6.2 e Tab. estabilidade do TCC).

Para cada par (modelo, imagem):
  - Carrega os heatmaps brutos .npy gerados por xai_generate_heatmaps.py
  - Calcula SSIM (Índice de Similaridade Estrutural) entre os dois mapas
  - Calcula Correlação de Pearson (ρ) entre os dois mapas linearizados

As métricas calculadas alimentam diretamente o eixo S_estab do ranking
multicritério (Sec. 3.7.4): S_estab = SSIM médio por modelo, normalizado
por min-max sobre os 12 modelos.

Saídas:
  xai_similarity_results.csv  — uma linha por (imagem, modelo)
  xai_similarity_summary.csv  — médias de SSIM e Pearson por modelo

Uso:
  python xai_similarity_analysis.py
         [--heatmaps xai_outputs]
         [--output_csv xai_similarity_results.csv]

Referência:
  Miranda, G.P. (2026). TCC — IFES Campus Serra.
"""

import argparse
import os
import warnings
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─── Constantes documentadas ──────────────────────────────────────────────────

# SSIM calculado sobre heatmaps contínuos normalizados [0, 1]
# data_range = 1.0 (valores já em [0,1])
SSIM_DATA_RANGE = 1.0


# ─── Carregamento ─────────────────────────────────────────────────────────────

def load_raw_heatmap(npy_path: str) -> np.ndarray:
    """
    Carrega um heatmap bruto salvo por xai_generate_heatmaps.py.
    Retorna array float32 normalizado [0, 1], shape = IMG_SIZE.
    """
    heatmap = np.load(npy_path).astype(np.float32)
    # Garante que está no intervalo [0, 1]
    hm_min, hm_max = heatmap.min(), heatmap.max()
    if hm_max - hm_min > 1e-8:
        heatmap = (heatmap - hm_min) / (hm_max - hm_min)
    return heatmap


# ─── Métricas de estabilidade ────────────────────────────────────────────────

def compute_ssim(grad_hm: np.ndarray, score_hm: np.ndarray) -> float:
    """
    SSIM entre o mapa Grad-CAM e o mapa Score-CAM do mesmo modelo
    sobre a mesma imagem (Sec. 3.6.2, Eq. SSIM).

    Ambos os mapas devem ter shape idêntico e valores em [0, 1].
    """
    if grad_hm.shape != score_hm.shape:
        import cv2
        score_hm = cv2.resize(
            score_hm,
            (grad_hm.shape[1], grad_hm.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return float(ssim(grad_hm, score_hm, data_range=SSIM_DATA_RANGE))


def compute_pearson(grad_hm: np.ndarray, score_hm: np.ndarray) -> float:
    """
    Correlação de Pearson (ρ) entre os vetores linearizados dos dois mapas.
    Retorna 0.0 se a correlação não puder ser calculada (variância nula).
    """
    x = grad_hm.flatten()
    y = score_hm.flatten()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    rho, _ = pearsonr(x, y)
    return float(rho)


# ─── Descoberta de pares (gradcam, scorecam) ──────────────────────────────────

def discover_pairs(heatmaps_root: str) -> list[dict]:
    """
    Percorre a estrutura de diretórios gerada por xai_generate_heatmaps.py:

      heatmaps_root/
        {model_name}/
          {image_id}_gradcam_raw.npy
          {image_id}_scorecam_raw.npy

    Retorna lista de dicts com chaves:
      model, image_id, gradcam_path, scorecam_path
    """
    grad_files = sorted(glob(
        os.path.join(heatmaps_root, "**", "*_gradcam_raw.npy"),
        recursive=True,
    ))

    pairs = []
    for grad_path in grad_files:
        p           = Path(grad_path)
        model_name  = p.parent.name
        image_id    = p.stem.replace("_gradcam_raw", "")
        score_path  = str(p).replace("_gradcam_raw.npy", "_scorecam_raw.npy")

        if not os.path.isfile(score_path):
            continue  # Score-CAM ausente para esta imagem — pular

        pairs.append({
            "model":        model_name,
            "image_id":     image_id,
            "gradcam_path": grad_path,
            "scorecam_path": score_path,
        })

    return pairs


# ─── Pipeline principal ───────────────────────────────────────────────────────

def analyze_similarity(heatmaps_root: str,
                       output_csv: str) -> pd.DataFrame:
    """
    Para cada par (model, image_id) calcula SSIM e Pearson entre
    os heatmaps Grad-CAM e Score-CAM.

    Exporta:
      output_csv             — resultados por imagem
      *_summary.csv          — médias por modelo (alimenta S_estab)
    """
    pairs = discover_pairs(heatmaps_root)

    if not pairs:
        raise FileNotFoundError(
            f"Nenhum par gradcam/scorecam encontrado em '{heatmaps_root}'. "
            "Execute xai_generate_heatmaps.py primeiro."
        )

    print(f"Pares (modelo, imagem) encontrados: {len(pairs)}")

    results = []
    for pair in tqdm(pairs, desc="Calculando similaridade", unit="par"):
        try:
            grad_hm  = load_raw_heatmap(pair["gradcam_path"])
            score_hm = load_raw_heatmap(pair["scorecam_path"])

            ssim_val    = compute_ssim(grad_hm, score_hm)
            pearson_val = compute_pearson(grad_hm, score_hm)

            results.append({
                "model":        pair["model"],
                "image_id":     pair["image_id"],
                "ssim":         round(ssim_val,    4),
                "pearson":      round(pearson_val, 4),
            })

        except Exception as exc:
            print(f"\n  [ERRO] {pair['model']} / {pair['image_id']}: {exc}")

    df = pd.DataFrame(results)

    # ── Exporta CSV completo ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResultados por imagem salvos em: {output_csv}")

    # ── Resumo por modelo (S_estab = SSIM médio) ───────────────────────────
    summary = (
        df.groupby("model")[["ssim", "pearson"]]
        .mean()
        .round(4)
        .reset_index()
        .sort_values("ssim", ascending=False)
    )

    summary_path = output_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Resumo por modelo    salvo em: {summary_path}")

    print("\n" + "="*60)
    print(" ESTABILIDADE INTERPRETATIVA (médias por modelo)")
    print(" Métrica S_estab = SSIM médio — alimenta o ranking (Sec. 3.7.4)")
    print("="*60)
    print(summary.to_string(index=False))

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análise de estabilidade interpretativa Grad-CAM vs Score-CAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--heatmaps", type=str, default="xai_outputs",
        help="Raiz com subdiretórios de heatmaps (*_raw.npy) por modelo",
    )
    parser.add_argument(
        "--output_csv", type=str, default="xai_similarity_results.csv",
        help="Caminho do CSV de saída completo",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.heatmaps):
        raise FileNotFoundError(
            f"Diretório de heatmaps não encontrado: {args.heatmaps}"
        )

    print("\nxai_retinopathy — Análise de Estabilidade Interpretativa")
    print(f"Heatmaps  : {os.path.abspath(args.heatmaps)}")
    print(f"Output CSV: {os.path.abspath(args.output_csv)}\n")

    analyze_similarity(args.heatmaps, args.output_csv)

    print("\n✔ Análise de estabilidade concluída.")