"""
training_hub.py
===============
Pipeline de treinamento para avaliação multicritério de CNNs aplicadas
ao diagnóstico de retinopatia diabética.

Executa os 24 experimentos documentados no trabalho:
  - 2 tarefas  × 4 arquiteturas × 3 pré-processamentos = 24 modelos
  - Tarefas: diagnóstico binário | classificação de severidade por agrupamento
  - Arquiteturas: DenseNet121, EfficientNetB3, ResNet50, VGG16
  - Pré-processamentos: Normalização Simples, CLAHE, Ben Graham Method

Referência:
  Miranda, G.P. (2026). Técnicas de Explicabilidade em Modelos de Redes Neurais
  Convolucionais para Diagnóstico da Retinopatia Diabética por Visão Computacional.
  TCC — IFES Campus Serra.

Uso:
  python training_hub.py --dataset <path> [--task diagnostico|classificacao|all]
                         [--arch <arch>] [--preprocessing <prep>] [--output <dir>]
"""

import argparse
import gc
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121, EfficientNetB3, ResNet50, VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─── Reprodutibilidade ────────────────────────────────────────────────────────

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ─── Configuração ────────────────────────────────────────────────────────────

class Config:
    """Hiperparâmetros e configurações do experimento (documentados no TCC, Sec. 3.5)."""

    # Treinamento
    BATCH_SIZE    = 32
    EPOCHS        = 100
    LR_PHASE1     = 1e-4   # Feature Extraction
    LR_PHASE2     = 1e-5   # Fine-tuning
    DROPOUT_1     = 0.5
    DROPOUT_2     = 0.3
    DENSE_UNITS   = 256

    # Early stopping (Sec. 3.5.2)
    ES_PATIENCE   = 15
    ES_MIN_DELTA  = 0.001

    # ReduceLROnPlateau (Sec. 3.5.2)
    LR_PATIENCE   = 5
    LR_FACTOR     = 0.5
    LR_MIN        = 1e-7

    # Particionamento (Sec. 3.3.1): 70 / 15 / 15
    VAL_TEST_SIZE = 0.30   # split inicial: 30% → depois 50/50 → 15% val + 15% test

    # Imagem
    IMG_SIZE      = (224, 224)

    # Fases de fine-tuning
    FINE_TUNE_LAYERS = 20  # últimas N camadas do backbone descongeladas na fase 2


# ─── Técnicas de pré-processamento (Sec. 3.4) ────────────────────────────────

class PreprocessingTechniques:
    """
    Três estratégias de pré-processamento avaliadas no trabalho.
    Todas retornam array float32 normalizado para [0, 1] com shape (H, W, 3).
    """

    @staticmethod
    def _to_rgb(img: np.ndarray) -> np.ndarray:
        """Garante formato RGB uint8."""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _scale_radius(img: np.ndarray, scale: int = 300) -> np.ndarray:
        """Normaliza o raio da retina para valor de referência fixo (Ben Graham)."""
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() // 2
        if r > 0:
            s = scale / r
            img = cv2.resize(img, (0, 0), fx=s, fy=s)
        return img

    @staticmethod
    def original(img: np.ndarray, final_size: tuple = Config.IMG_SIZE) -> np.ndarray:
        """
        Normalização Simples: apenas redimensionamento e normalização para [0, 1].
        Atua como baseline metodológico (Sec. 3.4.1).
        """
        img = PreprocessingTechniques._to_rgb(img)
        img = cv2.resize(img, final_size)
        return img.astype(np.float32) / 255.0

    @staticmethod
    def clahe(img: np.ndarray,
              clip_limit: float = 2.0,
              tile_grid_size: tuple = (8, 8),
              final_size: tuple = Config.IMG_SIZE) -> np.ndarray:
        """
        CLAHE: equalização adaptativa de histograma no canal L do espaço LAB.
        Parâmetros: clip_limit=2.0, tile_grid_size=(8,8) (Sec. 3.4.2).
        """
        img = PreprocessingTechniques._to_rgb(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe_obj.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        img_out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        img_out = cv2.resize(img_out, final_size)
        return img_out.astype(np.float32) / 255.0

    @staticmethod
    def ben_graham(img: np.ndarray,
                   scale: int = 300,
                   final_size: tuple = Config.IMG_SIZE) -> np.ndarray:
        """
        Ben Graham Method: normalização do raio + subtração gaussiana passa-alta
        + mascaramento circular (Sec. 3.4.3).

        Operação central: I' = 4I - 4*G_sigma(I) + 128
        onde sigma é proporcional ao raio estimado da retina.
        """
        img = PreprocessingTechniques._to_rgb(img)
        scaled = PreprocessingTechniques._scale_radius(img, scale)

        sigma = scale / 30
        blurred = cv2.GaussianBlur(scaled, (0, 0), sigma)
        processed = cv2.addWeighted(scaled, 4, blurred, -4, 128)

        # Mascaramento circular
        mask = np.zeros(processed.shape, dtype=np.uint8)
        cx, cy = processed.shape[1] // 2, processed.shape[0] // 2
        radius = int(scale * 0.9)
        cv2.circle(mask, (cx, cy), radius, (1, 1, 1), -1)
        processed = processed * mask + 128 * (1 - mask)

        processed = cv2.resize(processed, final_size)
        return np.clip(processed, 0, 255).astype(np.float32) / 255.0


PREPROCESSING_FUNCS = {
    "original":   PreprocessingTechniques.original,
    "clahe":      PreprocessingTechniques.clahe,
    "ben_graham": PreprocessingTechniques.ben_graham,
}


# ─── Carregamento de dados ────────────────────────────────────────────────────

def load_dataset(data_path: str, preprocess_fn) -> tuple:
    """
    Carrega imagens de um diretório organizado por subpastas de classe,
    aplica o pré-processamento e retorna arrays numpy.

    Estrutura esperada:
        data_path/
            classe_0/  (ex.: healthy)
            classe_1/  (ex.: disease)

    Retorno:
        (X, y, class_names) — arrays float32, int32, lista de strings
    """
    images, labels = [], []
    class_names = sorted([
        d for d in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, d))
    ])

    print(f"  Classes: {class_names}")

    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(data_path, cls)
        files = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
        ]
        print(f"    {cls}: {len(files)} imagens")

        for fname in files:
            fpath = os.path.join(cls_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"    [AVISO] Não foi possível ler: {fpath}")
                continue
            try:
                processed = preprocess_fn(img)
                images.append(processed)
                labels.append(idx)
            except Exception as exc:
                print(f"    [AVISO] Erro ao processar {fpath}: {exc}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, class_names


def split_dataset(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Particionamento estratificado 70 / 15 / 15 com seed fixo (Sec. 3.3.1).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=Config.VAL_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )
    print(f"  Split — treino: {len(X_train)}  val: {len(X_val)}  teste: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─── Arquitetura do modelo (Sec. 3.5.1) ──────────────────────────────────────

BACKBONE_MAP = {
    "densenet121":    DenseNet121,
    "efficientnetb3": EfficientNetB3,
    "resnet50":       ResNet50,
    "vgg16":          VGG16,
}


def build_model(architecture: str,
                input_shape: tuple = (*Config.IMG_SIZE, 3),
                num_classes: int = 2) -> tuple:
    """
    Constrói modelo com Transfer Learning (ImageNet).

    Estrutura: backbone (congelado) → GlobalAveragePooling2D
               → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3)
               → Dense(1, sigmoid)   [binary]
               → Dense(N, softmax)   [multiclass — não usado neste trabalho]

    Retorna (model, backbone) para permitir descongelamento na fase 2.
    """
    if architecture not in BACKBONE_MAP:
        raise ValueError(
            f"Arquitetura '{architecture}' não suportada. "
            f"Opções: {list(BACKBONE_MAP.keys())}"
        )

    backbone = BACKBONE_MAP[architecture](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(Config.DROPOUT_1)(x)
    x = layers.Dense(Config.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(Config.DROPOUT_2)(x)

    # Ambas as tarefas são binárias neste trabalho
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model, backbone


# ─── Treinamento em duas fases (Sec. 3.5.2) ──────────────────────────────────

def get_callbacks(model_path: str) -> list:
    return [
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=Config.ES_PATIENCE,
            min_delta=Config.ES_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=Config.LR_FACTOR,
            patience=Config.LR_PATIENCE,
            min_lr=Config.LR_MIN,
            verbose=1,
        ),
    ]


def get_augmentation() -> ImageDataGenerator:
    """
    Aumento de dados para imagens de fundo de retina.
    Rotação suave, pequenos deslocamentos, zoom moderado e flip horizontal.
    Flip vertical desativado (não aplicável a fundoscopia).
    """
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def train(model: keras.Model,
          backbone: keras.Model,
          X_train: np.ndarray, y_train: np.ndarray,
          X_val: np.ndarray,   y_val: np.ndarray,
          model_path: str,
          class_weights: dict) -> float:
    """
    Treinamento em duas fases para mitigar catastrophic forgetting (Sec. 3.5.2):
      Fase 1 — Feature Extraction: backbone congelado, lr = 1e-4, máx 20 épocas.
      Fase 2 — Fine-tuning: últimas N camadas descongeladas, lr = 1e-5, até completar 100.
    """
    datagen   = get_augmentation()
    callbacks = get_callbacks(model_path)
    t_start   = time.time()

    # ── Fase 1: Feature Extraction ─────────────────────────────────────────
    print("  Fase 1 — Feature Extraction")
    compile_model(model, Config.LR_PHASE1)

    model.fit(
        datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    # ── Fase 2: Fine-tuning ────────────────────────────────────────────────
    print(f"  Fase 2 — Fine-tuning (últimas {Config.FINE_TUNE_LAYERS} camadas)")
    backbone.trainable = True
    for layer in backbone.layers[: -Config.FINE_TUNE_LAYERS]:
        layer.trainable = False

    compile_model(model, Config.LR_PHASE2)
    callbacks = get_callbacks(model_path)  # reinicia callbacks para fase 2

    model.fit(
        datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS - 20,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    elapsed = time.time() - t_start
    print(f"  Treinamento concluído em {elapsed / 60:.1f} min")
    return elapsed


# ─── Avaliação (Sec. 3.6.1) ──────────────────────────────────────────────────

def evaluate(model: keras.Model,
             X_test: np.ndarray,
             y_test: np.ndarray,
             class_names: list,
             model_name: str,
             output_dir: str) -> dict:
    """
    Calcula ACC, Precisão, Sensibilidade, Especificidade, F1, AUC-ROC e Kappa.
    Gera e salva a matriz de confusão.
    """
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_proba > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc         = accuracy_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred, zero_division=0)
    sensitivity = recall_score(y_test, y_pred, zero_division=0)   # recall = sensibilidade
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(y_test, y_pred, zero_division=0)
    auc         = roc_auc_score(y_test, y_proba)
    kappa       = cohen_kappa_score(y_test, y_pred)

    print(f"\n  Resultados — {model_name}")
    print(f"    ACC={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  Kappa={kappa:.4f}")
    print(f"    Sens={sensitivity:.4f}  Espec={specificity:.4f}  Prec={precision:.4f}")
    print(f"    Matriz de confusão:\n{cm}")

    # Salva matriz de confusão
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 14}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"cm_{model_name}.png"), dpi=150)
    plt.close()

    return {
        "accuracy":    round(acc, 4),
        "precision":   round(precision, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_score":    round(f1, 4),
        "auc_roc":     round(auc, 4),
        "kappa":       round(kappa, 4),
    }


# ─── Experimento único ────────────────────────────────────────────────────────

def run_experiment(task: str,
                   architecture: str,
                   preprocessing: str,
                   dataset_root: str,
                   output_dir: str) -> dict:
    """
    Executa um experimento completo para uma combinação (tarefa, arquitetura, pré-proc).

    Args:
        task:          "diagnostico" | "classificacao"
        architecture:  "densenet121" | "efficientnetb3" | "resnet50" | "vgg16"
        preprocessing: "original" | "clahe" | "ben_graham"
        dataset_root:  raiz com subpastas model1_diagnostico/ e model2_classificacao/
        output_dir:    diretório de saída para modelos e resultados
    """
    model_name = f"{task}_{architecture}_{preprocessing}"
    print(f"\n{'='*65}")
    print(f" EXPERIMENTO: {model_name}")
    print(f"{'='*65}")

    # ── Caminhos ──────────────────────────────────────────────────────────
    folder_map = {
        "diagnostico":  "model1_diagnostico",
        "classificacao": "model2_classificacao",
    }
    data_path  = os.path.join(dataset_root, folder_map[task])
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.h5")

    # ── Carregamento e particionamento ────────────────────────────────────
    print("  Carregando dados...")
    preprocess_fn = PREPROCESSING_FUNCS[preprocessing]
    X, y, class_names = load_dataset(data_path, preprocess_fn)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    # ── Pesos de classe (desbalanceamento) ────────────────────────────────
    cw_array = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(cw_array))
    print(f"  Pesos de classe: {class_weights}")

    # ── Modelo ────────────────────────────────────────────────────────────
    model, backbone = build_model(
        architecture,
        input_shape=(*Config.IMG_SIZE, 3),
        num_classes=len(class_names),
    )
    print(f"  Parâmetros totais: {model.count_params():,}")

    # ── Treinamento ───────────────────────────────────────────────────────
    elapsed = train(
        model, backbone,
        X_train, y_train,
        X_val,   y_val,
        model_path,
        class_weights,
    )

    # ── Avaliação ─────────────────────────────────────────────────────────
    metrics = evaluate(model, X_test, y_test, class_names, model_name, output_dir)

    # ── Libera memória ────────────────────────────────────────────────────
    del model, backbone, X, X_train, X_val, X_test
    keras.backend.clear_session()
    gc.collect()

    return {
        "task":           task,
        "architecture":   architecture,
        "preprocessing":  preprocessing,
        "training_min":   round(elapsed / 60, 2),
        "model_path":     model_path,
        **metrics,
    }


# ─── Experimentos documentados (Sec. 4.1) ────────────────────────────────────

ALL_EXPERIMENTS = [
    # Tarefa 1 — Diagnóstico Binário (12 configurações)
    ("diagnostico", "densenet121",    "original"),
    ("diagnostico", "densenet121",    "clahe"),
    ("diagnostico", "densenet121",    "ben_graham"),
    ("diagnostico", "resnet50",       "original"),
    ("diagnostico", "resnet50",       "clahe"),
    ("diagnostico", "resnet50",       "ben_graham"),
    ("diagnostico", "efficientnetb3", "original"),
    ("diagnostico", "efficientnetb3", "clahe"),
    ("diagnostico", "efficientnetb3", "ben_graham"),
    ("diagnostico", "vgg16",          "original"),
    ("diagnostico", "vgg16",          "clahe"),
    ("diagnostico", "vgg16",          "ben_graham"),
    # Tarefa 2 — Classificação de Severidade por Agrupamento (12 configurações)
    ("classificacao", "densenet121",    "original"),
    ("classificacao", "densenet121",    "clahe"),
    ("classificacao", "densenet121",    "ben_graham"),
    ("classificacao", "resnet50",       "original"),
    ("classificacao", "resnet50",       "clahe"),
    ("classificacao", "resnet50",       "ben_graham"),
    ("classificacao", "efficientnetb3", "original"),
    ("classificacao", "efficientnetb3", "clahe"),
    ("classificacao", "efficientnetb3", "ben_graham"),
    ("classificacao", "vgg16",          "original"),
    ("classificacao", "vgg16",          "clahe"),
    ("classificacao", "vgg16",          "ben_graham"),
]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento — xai_retinopathy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Raiz do dataset (contém model1_diagnostico/ e model2_classificacao/)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Diretório de saída para modelos e resultados",
    )
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["diagnostico", "classificacao", "all"],
        help="Filtrar por tarefa",
    )
    parser.add_argument(
        "--arch", type=str, default="all",
        choices=["densenet121", "efficientnetb3", "resnet50", "vgg16", "all"],
        help="Filtrar por arquitetura",
    )
    parser.add_argument(
        "--preprocessing", type=str, default="all",
        choices=["original", "clahe", "ben_graham", "all"],
        help="Filtrar por técnica de pré-processamento",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    dataset_root = os.path.abspath(args.dataset)
    output_dir   = os.path.abspath(args.output)

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_root}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)

    print("\n" + "="*65)
    print(" xai_retinopathy — Training Hub")
    print("="*65)
    print(f" Dataset : {dataset_root}")
    print(f" Output  : {output_dir}")
    print(f" Task    : {args.task}")
    print(f" Arch    : {args.arch}")
    print(f" Prep    : {args.preprocessing}")
    print("="*65 + "\n")

    # Verifica GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU detectada: {gpus[0].name}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("AVISO: nenhuma GPU detectada. Treinamento será lento na CPU.")

    # Filtra experimentos conforme argumentos
    experiments = [
        (t, a, p) for t, a, p in ALL_EXPERIMENTS
        if (args.task == "all" or t == args.task)
        and (args.arch == "all" or a == args.arch)
        and (args.preprocessing == "all" or p == args.preprocessing)
    ]

    print(f"Experimentos a executar: {len(experiments)} / {len(ALL_EXPERIMENTS)}\n")

    results = []
    results_path = os.path.join(output_dir, "results", "training_results.csv")

    for idx, (task, arch, prep) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}]")
        try:
            result = run_experiment(task, arch, prep, dataset_root, output_dir)
            results.append(result)

            # Salva resultados parciais a cada experimento
            pd.DataFrame(results).to_csv(results_path, index=False)
            print(f"  Resultados salvos em: {results_path}")

        except Exception as exc:
            print(f"  ERRO no experimento ({task}, {arch}, {prep}): {exc}")
            results.append({
                "task": task, "architecture": arch, "preprocessing": prep,
                "error": str(exc),
            })

    # Resumo final
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*65)
        print(" RESUMO FINAL")
        print("="*65)
        cols = ["task", "architecture", "preprocessing", "auc_roc", "accuracy", "kappa"]
        available = [c for c in cols if c in df.columns]
        print(df[available].to_string(index=False))
        print(f"\nResultados completos: {results_path}")


if __name__ == "__main__":
    main()