"""
xai_generate_heatmaps.py
========================
Geração dos mapas de ativação Grad-CAM e Score-CAM para todos os modelos
treinados, sobre o conjunto de teste e sobre as imagens do DIARETDB1.

Documenta a Seção 3.5 do TCC:
  - Seleção automática da última camada Conv2D (compatível com todas as arquiteturas)
  - Score-CAM com os 128 canais de maior ativação média
  - Colormap TURBO, alpha de overlay = 0.48
  - Saída: xai_outputs/{model_name}/{image_id}_gradcam.png
                                      {image_id}_scorecam.png
                                      {image_id}_gradcam_raw.npy
                                      {image_id}_scorecam_raw.npy  ← para avaliação quantitativa

Uso:
  python xai_generate_heatmaps.py --models outputs/models
                                  --images ddb1_fundusimages
                                  [--output xai_outputs]
                                  [--max_channels 128]

Referência:
  Miranda, G.P. (2026). TCC — IFES Campus Serra.
"""

import argparse
import os
import random
import warnings
from glob import glob
from pathlib import Path

import cv2
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import (
    DenseNet121, EfficientNetB3, ResNet50, VGG16
)
from tensorflow.keras.models import model_from_json
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ─── Reprodutibilidade ────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Constantes documentadas (Sec. 3.5) ──────────────────────────────────────

IMG_SIZE      = (224, 224)   # tamanho de entrada documentado
MAX_CHANNELS  = 128          # Score-CAM: 128 canais de maior ativação média
OVERLAY_ALPHA = 0.48         # peso do heatmap no blend com a imagem original

# ─── Mapa de arquiteturas (deve coincidir com training_hub.py) ─────────────────

BACKBONE_MAP = {
    "densenet121":    DenseNet121,
    "efficientnetb3": EfficientNetB3,
    "resnet50":       ResNet50,
    "vgg16":          VGG16,
}


# ─── Carregamento de modelos ──────────────────────────────────────────────────

def _infer_architecture(model_path: str) -> str | None:
    """Infere a arquitetura pelo nome do arquivo de modelo."""
    stem = Path(model_path).stem.lower()
    for arch in BACKBONE_MAP:
        if arch in stem:
            return arch
    return None


def _build_model_from_weights(weights_path: str, architecture: str) -> keras.Model:
    """
    Reconstrói a arquitetura documentada no TCC e carrega os pesos.
    Usado como fallback quando load_model falha por incompatibilidade Keras.

    Estrutura: backbone → GAP → Dropout(0.5) → Dense(256) → Dropout(0.3) → Dense(1, sigmoid)
    """
    backbone_fn = BACKBONE_MAP[architecture]
    backbone = backbone_fn(
        include_top=False,
        weights=None,
        input_shape=(*IMG_SIZE, 3),
    )
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(backbone.input, outputs)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return model


def load_model(model_path: str) -> keras.Model:
    """
    Carrega um modelo .h5 com tratamento de incompatibilidades
    entre Keras 2.x e Keras 3.

    Estratégia:
      1. Tentativa padrão (keras.models.load_model)
      2. Modo compatível: lê model_config do HDF5, corrige schema, reconstrói
      3. Fallback: reconstrói arquitetura do zero e carrega apenas os pesos
    """
    # ── Tentativa 1: load padrão ──────────────────────────────────────────
    try:
        model = keras.models.load_model(model_path, compile=False)
        model.trainable = False
        return model
    except Exception as e1:
        print(f"  [AVISO] Load padrão falhou: {e1}")

    # ── Tentativa 2: modo compatível via HDF5 ─────────────────────────────
    try:
        with h5py.File(model_path, "r") as f:
            cfg = f.attrs.get("model_config")
            if cfg is None:
                raise ValueError("model_config não encontrado no HDF5.")
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8")
            cfg = cfg.replace('"batch_shape"', '"batch_input_shape"')
            model = model_from_json(cfg)
            model.load_weights(model_path)
        model.trainable = False
        print("  [INFO] Modelo carregado em modo compatível.")
        return model
    except Exception as e2:
        print(f"  [AVISO] Modo compatível falhou: {e2}")

    # ── Tentativa 3: reconstrução da arquitetura + pesos ──────────────────
    arch = _infer_architecture(model_path)
    if arch is None:
        raise RuntimeError(
            f"Não foi possível carregar '{model_path}'. "
            f"Verifique se o nome do arquivo contém a arquitetura "
            f"(ex.: 'diagnostico_resnet50_clahe.h5')."
        )
    print(f"  [INFO] Reconstruindo arquitetura '{arch}' e carregando pesos.")
    model = _build_model_from_weights(model_path, arch)
    model.trainable = False
    return model


# ─── Carregamento de imagens ──────────────────────────────────────────────────

def load_image(img_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (batch_array, original_uint8):
      batch_array  — shape (1, H, W, 3), float32 normalizado [0, 1]
      original     — shape (H, W, 3), uint8 RGB para visualização
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, IMG_SIZE)
    batch = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return batch, resized


# ─── Seleção automática da camada convolucional (Sec. 3.5) ───────────────────

def find_last_conv_layer(model: keras.Model) -> str:
    """
    Percorre as camadas em ordem reversa e retorna o nome da última
    instância de Conv2D — compatível com todas as arquiteturas avaliadas
    sem necessidade de especificação manual (Sec. 3.5).
    """
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError(
        "Nenhuma camada Conv2D encontrada no modelo. "
        "Verifique se o modelo é uma CNN."
    )


# ─── Grad-CAM (Sec. 3.5.1 / Eq. gradcam_weights e gradcam_map) ──────────────

def gradcam(model: keras.Model,
            img_array: np.ndarray,
            layer_name: str) -> np.ndarray:
    """
    Implementa Grad-CAM conforme Selvaraju et al. (2017).

    Para classificação binária com ativação sigmoid (saída escalar),
    o gradiente é calculado diretamente em relação à saída do modelo
    (sem argmax, pois há apenas uma classe positiva).

    Retorna heatmap normalizado [0, 1] com shape (h_conv, w_conv).
    """
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        # Saída binária sigmoid: usar o score diretamente como loss
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)                   # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (c,)

    conv_out = conv_outputs[0]                                   # (h, w, c)
    heatmap  = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (h, w)
    heatmap  = tf.maximum(heatmap, 0)                           # ReLU
    heatmap  = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ─── Score-CAM (Sec. 3.5.2 / Eq. scorecam_weights e scorecam_map) ────────────

def scorecam(model: keras.Model,
             img_array: np.ndarray,
             layer_name: str,
             max_channels: int = MAX_CHANNELS) -> np.ndarray:
    """
    Implementa Score-CAM conforme Wang et al. (2020).

    Seleciona os `max_channels` canais com maior ativação média para
    limitar o custo computacional (O(max_channels) forward passes).

    Retorna heatmap normalizado [0, 1] com shape IMG_SIZE.
    """
    activation_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output,
    )

    activations = activation_model(img_array, training=False)[0].numpy()  # (h, w, c)

    # Seleciona os max_channels canais de maior ativação média
    channel_means = np.mean(activations, axis=(0, 1))
    top_idx = np.argsort(channel_means)[-max_channels:]

    heatmap = np.zeros(IMG_SIZE, dtype=np.float32)
    baseline_score = float(
        model(np.zeros_like(img_array), training=False)[0, 0]
    )

    for ch_idx in top_idx:
        act = activations[:, :, ch_idx].astype(np.float32)
        act_resized = cv2.resize(act, IMG_SIZE)

        # Normalização para [0, 1] como máscara multiplicativa
        act_min, act_max = act_resized.min(), act_resized.max()
        if act_max - act_min < 1e-8:
            continue
        mask = (act_resized - act_min) / (act_max - act_min)
        mask = mask[..., np.newaxis]                    # (H, W, 1)

        masked_input = img_array * mask                 # (1, H, W, 3)

        score = float(model(masked_input, training=False)[0, 0])
        weight = score - baseline_score                 # impacto relativo

        heatmap += weight * mask[..., 0]

    heatmap = np.maximum(heatmap, 0)                    # ReLU
    hm_max = heatmap.max()
    if hm_max > 1e-8:
        heatmap /= hm_max

    return heatmap


# ─── Overlay (colormap TURBO, documentado no TCC) ────────────────────────────

def overlay_heatmap(heatmap: np.ndarray,
                    original_rgb: np.ndarray,
                    alpha: float = OVERLAY_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona o heatmap para o tamanho da imagem original,
    aplica colormap TURBO e combina com a imagem por blending linear.

    alpha = peso do heatmap no blend (documentado como 0.48).

    Retorna (colored_heatmap_bgr, overlay_rgb).
    """
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_u8 = np.uint8(255 * heatmap_resized)
    colored_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)

    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr  = cv2.addWeighted(original_bgr, 1 - alpha, colored_bgr, alpha, 0)
    overlay_rgb  = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return colored_bgr, overlay_rgb


# ─── Salvamento ───────────────────────────────────────────────────────────────

def save_outputs(out_dir: str,
                 image_id: str,
                 grad_overlay: np.ndarray,
                 score_overlay: np.ndarray,
                 grad_raw: np.ndarray,
                 score_raw: np.ndarray) -> None:
    """
    Salva overlays (PNG) e heatmaps brutos (NPY) para uso pela avaliação quantitativa.

    Estrutura:
      out_dir/
        {image_id}_gradcam.png
        {image_id}_scorecam.png
        {image_id}_gradcam_raw.npy    ← heatmap normalizado [0,1], shape IMG_SIZE
        {image_id}_scorecam_raw.npy   ← idem
    """
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(
        os.path.join(out_dir, f"{image_id}_gradcam.png"),
        cv2.cvtColor(grad_overlay, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{image_id}_scorecam.png"),
        cv2.cvtColor(score_overlay, cv2.COLOR_RGB2BGR),
    )
    np.save(os.path.join(out_dir, f"{image_id}_gradcam_raw.npy"),  grad_raw)
    np.save(os.path.join(out_dir, f"{image_id}_scorecam_raw.npy"), score_raw)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def process_model(model_path: str,
                  image_dir: str,
                  output_root: str,
                  max_channels: int = MAX_CHANNELS) -> None:
    """
    Executa o pipeline completo de geração de heatmaps para um modelo:
      1. Carrega o modelo
      2. Identifica a última camada Conv2D
      3. Para cada imagem em image_dir: gera Grad-CAM e Score-CAM, salva resultados

    Saídas em: output_root/{model_name}/{image_id}_*.png / *.npy
    """
    model_name = Path(model_path).stem
    out_dir    = os.path.join(output_root, model_name)

    print(f"\n{'='*60}")
    print(f" Modelo: {model_name}")
    print(f" Saída:  {out_dir}")
    print(f"{'='*60}")

    model      = load_model(model_path)
    layer_name = find_last_conv_layer(model)
    print(f" Camada CAM: {layer_name}")

    images = sorted(
        glob(os.path.join(image_dir, "*.png")) +
        glob(os.path.join(image_dir, "*.jpg")) +
        glob(os.path.join(image_dir, "*.tif"))
    )

    if not images:
        print(f"  [AVISO] Nenhuma imagem encontrada em: {image_dir}")
        return

    print(f" Imagens a processar: {len(images)}\n")

    for img_path in tqdm(images, desc=model_name, unit="img"):
        image_id = Path(img_path).stem

        try:
            img_array, original = load_image(img_path)

            grad_raw  = gradcam(model, img_array, layer_name)
            score_raw = scorecam(model, img_array, layer_name, max_channels)

            _, grad_overlay  = overlay_heatmap(grad_raw,  original)
            _, score_overlay = overlay_heatmap(score_raw, original)

            save_outputs(
                out_dir, image_id,
                grad_overlay, score_overlay,
                grad_raw, score_raw,
            )

        except Exception as exc:
            print(f"\n  [ERRO] {image_id}: {exc}")

    print(f"\n  ✔ {len(images)} imagens processadas → {out_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geração de heatmaps Grad-CAM e Score-CAM — xai_retinopathy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", type=str, required=True,
        help="Diretório contendo os modelos .h5 treinados (ex.: outputs/models/)",
    )
    parser.add_argument(
        "--images", type=str, required=True,
        help="Diretório de imagens para gerar os heatmaps (ex.: ddb1_fundusimages/)",
    )
    parser.add_argument(
        "--output", type=str, default="xai_outputs",
        help="Diretório raiz de saída",
    )
    parser.add_argument(
        "--max_channels", type=int, default=MAX_CHANNELS,
        help="Número máximo de canais para Score-CAM",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Filtro opcional por substring no nome do modelo (ex.: 'resnet50')",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    models_dir = os.path.abspath(args.models)
    image_dir  = os.path.abspath(args.images)
    output_dir = os.path.abspath(args.output)

    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Diretório de modelos não encontrado: {models_dir}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")

    model_files = sorted(
        glob(os.path.join(models_dir, "*.h5")) +
        glob(os.path.join(models_dir, "*.keras"))
    )

    if args.filter:
        model_files = [m for m in model_files if args.filter.lower() in m.lower()]

    if not model_files:
        raise FileNotFoundError(
            f"Nenhum modelo encontrado em '{models_dir}'"
            + (f" com filtro '{args.filter}'" if args.filter else "")
        )

    print(f"\nxai_retinopathy — Geração de Heatmaps")
    print(f"Modelos encontrados : {len(model_files)}")
    print(f"Diretório de imagens: {image_dir}")
    print(f"Saída               : {output_dir}")
    print(f"max_channels (SCAM) : {args.max_channels}")

    for model_path in model_files:
        process_model(
            model_path=model_path,
            image_dir=image_dir,
            output_root=output_dir,
            max_channels=args.max_channels,
        )

    print("\n✔ Pipeline concluído.")


if __name__ == "__main__":
    main()