"""
xai_quantitative_evaluation.py
===============================
Cálculo das métricas espaciais de sobreposição entre os mapas de ativação
binarizados e as máscaras GT do DIARETDB1 (Sec. 3.6.3 do TCC).

Para cada par (modelo, imagem):
  - Carrega o heatmap bruto .npy gerado por xai_generate_heatmaps.py
  - Binariza pelo percentil 85 sobre os pixels da retina (Sec. 3.6.3)
  - Carrega a máscara GT = união lógica das 4 camadas de lesão do DIARETDB1
  - Calcula IoU, Precisão, Recall e F1-score espaciais
  - Salva visualização da sobreposição TP/FP/FN

Saídas:
  xai_quantitative_results.csv  — uma linha por (imagem, modelo, técnica CAM)
  xai_summary.csv               — médias por (modelo, técnica CAM)
  visual_outputs/{model}/{image_id}_gradcam_comparison.png
  visual_outputs/{model}/{image_id}_scorecam_comparison.png

Uso:
  python xai_quantitative_evaluation.py
         --heatmaps xai_outputs
         --groundtruth ddb1_groundtruth
         --fundus ddb1_fundusimages
         [--output_csv xai_quantitative_results.csv]
         [--visual_dir visual_outputs]

Referência:
  Miranda, G.P. (2026). TCC — IFES Campus Serra.
"""

import argparse
import os
import random
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Reprodutibilidade ────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Constantes documentadas (Sec. 3.6.3) ────────────────────────────────────

IMG_SIZE             = (224, 224)   # tamanho de entrada documentado
PERCENTILE_THRESHOLD = 85           # binarização do heatmap

LESION_TYPES = [
    "hardexudates",
    "hemorrhages",
    "softexudates",
    "redsmalldots",
]


# ─── Carregamento de máscaras GT ─────────────────────────────────────────────

def load_single_mask(path: str, size: tuple = IMG_SIZE) -> np.ndarray:
    """Carrega uma máscara de lesão em escala de cinza e binariza."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Máscara não encontrada: {path}")
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def load_groundtruth(image_id: str, gt_root: str,
                     size: tuple = IMG_SIZE) -> np.ndarray:
    """
    Carrega a máscara GT como união lógica das 4 camadas de lesão
    anotadas no DIARETDB1 (Eq. GT(x,y) = ∪_k M_k(x,y), Sec. 3.6.3).

    Retorna array binário uint8, shape = size.
    """
    combined = np.zeros(size, dtype=np.uint8)
    for lesion in LESION_TYPES:
        files = glob(os.path.join(gt_root, lesion, f"{image_id}*"))
        for fpath in files:
            combined = np.logical_or(combined, load_single_mask(fpath, size))
    return combined.astype(np.uint8)


# ─── Carregamento de imagem original ─────────────────────────────────────────

def load_original(image_id: str, fundus_dir: str) -> np.ndarray:
    """Carrega a imagem original de retina em RGB."""
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        path = os.path.join(fundus_dir, image_id + ext)
        if os.path.exists(path):
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(
        f"Imagem original não encontrada para '{image_id}' em '{fundus_dir}'"
    )


# ─── Binarização do heatmap (percentil 85) ───────────────────────────────────

def binarize_heatmap(heatmap: np.ndarray,
                     percentile: int = PERCENTILE_THRESHOLD) -> np.ndarray:
    """
    Binariza o heatmap normalizado [0,1] pelo percentil `percentile`
    calculado sobre todos os pixels (Sec. 3.6.3).

    Recebe array float32 de shape IMG_SIZE, retorna array uint8 binário.
    """
    if heatmap.shape != IMG_SIZE:
        heatmap = cv2.resize(
            heatmap.astype(np.float32), IMG_SIZE,
            interpolation=cv2.INTER_LINEAR
        )
    threshold = np.percentile(heatmap, percentile)
    return (heatmap >= threshold).astype(np.uint8)


# ─── Métricas espaciais ───────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray,
                    gt: np.ndarray) -> dict[str, float]:
    """
    Calcula IoU, Precisão, Recall e F1-score espaciais entre
    mapa de ativação binarizado (pred) e máscara GT.

    Ambos os arrays devem ser binários uint8 de mesmo shape.
    """
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    union = tp + fp + fn

    iou       = tp / union          if union > 0           else 0.0
    precision = tp / (tp + fp)      if (tp + fp) > 0       else 0.0
    recall    = tp / (tp + fn)      if (tp + fn) > 0       else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "iou":       round(float(iou),       4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall),    4),
        "f1":        round(float(f1),        4),
    }


# ─── Visualização TP/FP/FN ───────────────────────────────────────────────────

def save_visual_comparison(original_rgb: np.ndarray,
                            pred: np.ndarray,
                            gt: np.ndarray,
                            save_path: str) -> None:
    """
    Gera e salva visualização colorida da sobreposição:
      Verde  — TP (interseção entre pred e GT)
      Vermelho — FP (pred ativado fora da lesão)
      Amarelo  — FN (lesão não coberta pelo CAM)
      Branco   — contorno da máscara GT
    """
    h, w = original_rgb.shape[:2]

    pred_r = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    gt_r   = cv2.resize(gt.astype(np.uint8),   (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = original_rgb.copy().astype(np.float32)

    tp_mask = (pred_r == 1) & (gt_r == 1)
    fp_mask = (pred_r == 1) & (gt_r == 0)
    fn_mask = (pred_r == 0) & (gt_r == 1)

    alpha = 0.55
    overlay[tp_mask] = (1 - alpha) * overlay[tp_mask] + alpha * np.array([0,   200, 50])
    overlay[fp_mask] = (1 - alpha) * overlay[fp_mask] + alpha * np.array([220, 30,  30])
    overlay[fn_mask] = (1 - alpha) * overlay[fn_mask] + alpha * np.array([255, 215, 0 ])

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Contorno GT em branco
    contours, _ = cv2.findContours(
        gt_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# ─── Pipeline principal ───────────────────────────────────────────────────────

def evaluate(heatmaps_root: str,
             gt_root: str,
             fundus_dir: str,
             output_csv: str,
             visual_dir: str) -> pd.DataFrame:
    """
    Percorre a estrutura de saída do xai_generate_heatmaps.py:

      heatmaps_root/
        {model_name}/
          {image_id}_gradcam_raw.npy
          {image_id}_scorecam_raw.npy
          ...

    Para cada par (model, image_id, cam_type) calcula as métricas
    espaciais e salva a visualização.
    """
    # Descobre todos os heatmaps brutos
    npy_files = sorted(glob(
        os.path.join(heatmaps_root, "**", "*_raw.npy"),
        recursive=True,
    ))

    if not npy_files:
        raise FileNotFoundError(
            f"Nenhum arquivo *_raw.npy encontrado em '{heatmaps_root}'. "
            "Execute xai_generate_heatmaps.py primeiro."
        )

    print(f"Heatmaps encontrados : {len(npy_files)}")
    print(f"GT root              : {gt_root}")
    print(f"Fundus dir           : {fundus_dir}\n")

    results     = []
    skipped_gt  = 0

    for npy_path in tqdm(npy_files, desc="Avaliando", unit="heatmap"):
        p           = Path(npy_path)
        model_name  = p.parent.name                  # diretório pai = nome do modelo
        stem        = p.stem                          # ex.: image001_gradcam_raw
        # stem termina em _gradcam_raw ou _scorecam_raw
        cam_type    = stem.split("_")[-2]             # "gradcam" ou "scorecam"
        image_id    = stem.replace(f"_{cam_type}_raw", "")

        # ── GT ────────────────────────────────────────────────────────────
        gt = load_groundtruth(image_id, gt_root)
        if gt.sum() == 0:
            skipped_gt += 1
            continue

        # ── Heatmap ────────────────────────────────────────────────────────
        heatmap = np.load(npy_path).astype(np.float32)
        pred    = binarize_heatmap(heatmap)

        # ── Métricas ───────────────────────────────────────────────────────
        metrics = compute_metrics(pred, gt)

        results.append({
            "image":      image_id,
            "model":      model_name,
            "cam":        cam_type,
            **metrics,
        })

        # ── Visualização ───────────────────────────────────────────────────
        try:
            original = load_original(image_id, fundus_dir)
            vis_path = os.path.join(
                visual_dir, model_name,
                f"{image_id}_{cam_type}_comparison.png",
            )
            save_visual_comparison(original, pred, gt, vis_path)
        except FileNotFoundError:
            pass   # imagem original ausente — não bloqueia as métricas

    if skipped_gt > 0:
        print(f"\n[INFO] {skipped_gt} imagens ignoradas (GT vazio — sem lesões anotadas)")

    df = pd.DataFrame(results)

    # ── Exporta CSV completo ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResultados completos salvos em: {output_csv}")

    # ── Resumo por (model, cam) ────────────────────────────────────────────
    summary_path = os.path.join(
        os.path.dirname(os.path.abspath(output_csv)), "xai_summary.csv"
    )
    summary = (
        df.groupby(["model", "cam"])[["iou", "precision", "recall", "f1"]]
        .mean()
        .round(4)
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)

    print(f"Resumo por modelo    salvo em: {summary_path}")
    print("\n" + "="*65)
    print(" RESUMO — Médias por Modelo e Técnica CAM")
    print("="*65)
    print(summary.to_string(index=False))

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliação quantitativa XAI — xai_retinopathy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--heatmaps", type=str, default="xai_outputs",
        help="Raiz com subdiretórios de heatmaps (*_raw.npy) por modelo",
    )
    parser.add_argument(
        "--groundtruth", type=str, default="ddb1_groundtruth",
        help="Raiz das máscaras GT do DIARETDB1",
    )
    parser.add_argument(
        "--fundus", type=str, default="ddb1_fundusimages",
        help="Diretório com imagens de retina originais do DIARETDB1",
    )
    parser.add_argument(
        "--output_csv", type=str, default="xai_quantitative_results.csv",
        help="Caminho do CSV de saída completo",
    )
    parser.add_argument(
        "--visual_dir", type=str, default="visual_outputs",
        help="Diretório para imagens de visualização TP/FP/FN",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    for path, label in [
        (args.heatmaps,    "heatmaps"),
        (args.groundtruth, "groundtruth"),
        (args.fundus,      "fundus"),
    ]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Diretório '{label}' não encontrado: {path}")

    print("\nxai_retinopathy — Avaliação Quantitativa")
    print(f"Heatmaps   : {os.path.abspath(args.heatmaps)}")
    print(f"GT root    : {os.path.abspath(args.groundtruth)}")
    print(f"Fundus     : {os.path.abspath(args.fundus)}")
    print(f"Output CSV : {os.path.abspath(args.output_csv)}")
    print(f"Visual dir : {os.path.abspath(args.visual_dir)}\n")

    evaluate(
        heatmaps_root=args.heatmaps,
        gt_root=args.groundtruth,
        fundus_dir=args.fundus,
        output_csv=args.output_csv,
        visual_dir=args.visual_dir,
    )

    print("\n✔ Avaliação concluída.")


if __name__ == "__main__":
    main()