"""
train_transformers_comparison.py
=================================
Script untuk melatih, mengevaluasi, dan membandingkan beberapa model berbasis Transformer
(XLM-RoBERTa, IndoBERT, mBERT, IndoLEM IndoBERT) pada dataset cyberbullying bilingual.

Model yang dibandingkan:
  1. XLM-RoBERTa Base (xlm-roberta-base) — Multilingual Transformer (Model Utama)
  2. IndoBERT Base (indobenchmark/indobert-base-p1) — Monolingual Indonesian Transformer
  3. mBERT Base (bert-base-multilingual-cased) — Multilingual BERT
  4. IndoBERT Uncased (indolem/indobert-base-uncased) — Monolingual Indonesian Transformer (IndoLEM)

Cara jalankan:
    python src/models/train_transformers_comparison.py

Output:
    outputs/transformer_comparison/
        transformer_report.txt
        transformer_results.json
        transformer_f1_comparison.png
        confusion_matrices_transformers.png
"""

import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed as hf_set_seed,
)

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("transformer_comparison")

# ──────────────────────────────────────────────────────────────
# Paths & Config
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transformer_comparison"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "transformer_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
RANDOM_SEED = 42
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16_ENABLED = torch.cuda.is_available()

# Daftar Model Transformer yang akan dibandingkan
MODELS_TO_COMPARE = [
    {
        "key": "xlm-roberta",
        "name": "XLM-RoBERTa Base",
        "hf_id": "xlm-roberta-base",
        "type": "Multilingual",
        "pretrained_path": PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model",
    },
    {
        "key": "indobert-p1",
        "name": "IndoBERT Base (IndoBenchmark)",
        "hf_id": "indobenchmark/indobert-base-p1",
        "type": "Indonesian Monolingual",
        "pretrained_path": None,
    },
    {
        "key": "mbert-cased",
        "name": "mBERT Base (Multilingual)",
        "hf_id": "bert-base-multilingual-cased",
        "type": "Multilingual",
        "pretrained_path": None,
    },
    {
        "key": "indobert-indolem",
        "name": "IndoBERT Uncased (IndoLEM)",
        "hf_id": "indolem/indobert-base-uncased",
        "type": "Indonesian Monolingual",
        "pretrained_path": None,
    },
]


# ──────────────────────────────────────────────────────────────
# Seed Global
# ──────────────────────────────────────────────────────────────
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🌱 Global seed di-set ke: {seed}")


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────
class CyberbullyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label,
        }


# ──────────────────────────────────────────────────────────────
# Weighted Loss Trainer
# ──────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────
# Compute Metrics Function
# ──────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_wt, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "precision_binary": round(p_bin, 4),
        "recall_binary": round(r_bin, 4),
        "f1_binary": round(f1_bin, 4),
        "precision_macro": round(p_mac, 4),
        "recall_macro": round(r_mac, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
    }


def evaluate_with_threshold(logits, labels, threshold=0.5):
    probs = softmax(logits, axis=1)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": float(acc),
        "precision_binary": float(p_bin),
        "recall_binary": float(r_bin),
        "f1_binary": float(f1_bin),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f1_mac),
        "confusion_matrix": cm.tolist(),
    }


# ──────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────
def main():
    set_all_seeds(RANDOM_SEED)

    logger.info("=" * 70)
    logger.info("🚀 BENCHMARK MODEL TRANSFORMER: CYBERBULLYING DETECTION")
    logger.info("=" * 70)

    # 1. Load Data
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    logger.info(f"Data ukuran: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Hitung class weights
    classes = np.unique(train_df["label"])
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["label"].values
    )
    class_weights = torch.tensor(weights, dtype=torch.float)
    logger.info(f"Class weights (0, 1): {weights.tolist()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} (FP16={FP16_ENABLED})")

    results_all = []

    for model_info in MODELS_TO_COMPARE:
        key = model_info["key"]
        name = model_info["name"]
        hf_id = model_info["hf_id"]
        model_type = model_info["type"]
        pretrained_path = model_info["pretrained_path"]

        logger.info("\n" + "─" * 70)
        logger.info(f"📦 Processing: {name} [{hf_id}] ({model_type})")
        logger.info("─" * 70)

        output_model_dir = MODEL_SAVE_DIR / key
        best_model_path = output_model_dir / "best_model"

        start_time = time.time()

        # Cek jika model sudah pernah di-train/ada pretrained checkpoint
        if pretrained_path and pretrained_path.exists():
            logger.info(f"✅ Pretrained model ditemukan di: {pretrained_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(pretrained_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(pretrained_path))
            train_time = 0.0
        elif best_model_path.exists():
            logger.info(f"✅ Saved model ditemukan di: {best_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(best_model_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(best_model_path))
            train_time = 0.0
        else:
            logger.info(f"📥 Memuat tokenizer & base model dari HuggingFace: {hf_id}")
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_id, num_labels=2
            )

            # PyTorch Datasets
            train_dataset = CyberbullyDataset(
                train_df["text"], train_df["label"], tokenizer, MAX_LENGTH
            )
            val_dataset = CyberbullyDataset(
                val_df["text"], val_df["label"], tokenizer, MAX_LENGTH
            )

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            training_args = TrainingArguments(
                output_dir=str(output_model_dir / "checkpoints"),
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE_TRAIN,
                per_device_eval_batch_size=BATCH_SIZE_EVAL,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                warmup_ratio=WARMUP_RATIO,
                fp16=FP16_ENABLED,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                logging_steps=50,
                report_to="none",
                seed=RANDOM_SEED,
            )

            trainer = WeightedTrainer(
                class_weights=class_weights,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )

            logger.info(f"🏋️ Start fine-tuning {name} ...")
            t0 = time.time()
            trainer.train()
            train_time = round(time.time() - t0, 2)
            logger.info(f"⏱️ Training selasai dalam {train_time}s")

            # Save best model
            best_model_path.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            logger.info(f"💾 Best model disimpan di: {best_model_path}")

        # ──────────────────────────────────────────────────────────
        # EVALUATION
        # ──────────────────────────────────────────────────────────
        model.to(device)
        model.eval()

        val_dataset = CyberbullyDataset(val_df["text"], val_df["label"], tokenizer, MAX_LENGTH)
        test_dataset = CyberbullyDataset(test_df["text"], test_df["label"], tokenizer, MAX_LENGTH)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        eval_trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(PROJECT_ROOT / "models" / "temp"),
                per_device_eval_batch_size=BATCH_SIZE_EVAL,
                fp16=FP16_ENABLED,
                report_to="none",
            ),
            data_collator=data_collator,
        )

        logger.info(f"🔍 Evaluating on Validation Set ...")
        val_pred = eval_trainer.predict(val_dataset)
        val_metrics = evaluate_with_threshold(val_pred.predictions, val_df["label"].values, threshold=0.5)

        logger.info(f"🔍 Evaluating on Test Set ...")
        t_infer0 = time.time()
        test_pred = eval_trainer.predict(test_dataset)
        infer_time = round(time.time() - t_infer0, 4)
        samples_per_sec = round(len(test_df) / max(infer_time, 1e-4), 2)
        test_metrics = evaluate_with_threshold(test_pred.predictions, test_df["label"].values, threshold=0.5)

        res_entry = {
            "key": key,
            "name": name,
            "hf_id": hf_id,
            "type": model_type,
            "train_time_seconds": train_time,
            "infer_time_seconds": infer_time,
            "samples_per_second": samples_per_sec,
            "validation": val_metrics,
            "test": test_metrics,
        }
        results_all.append(res_entry)

        logger.info(
            f"  Val F1-Macro: {val_metrics['f1_macro']:.4f} | "
            f"Test F1-Macro: {test_metrics['f1_macro']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f}"
        )

    # ──────────────────────────────────────────────────────────
    # SAVE & REPORT
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("📊 METRAPKAN HASIL KOMPARASI DAN LAPORAN")
    logger.info("=" * 70)

    # 1. JSON Output
    json_path = OUTPUT_DIR / "transformer_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2)
    logger.info(f"💾 JSON disimpan: {json_path}")

    # 2. Text Report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LAPORAN PERBANDINGAN MODEL TRANSFORMER — CYBERBULLYING DETECTION")
    report_lines.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    for split in ["validation", "test"]:
        report_lines.append(f"\n{'─' * 80}")
        report_lines.append(f"  {split.upper()} SET RESULTS")
        report_lines.append(f"{'─' * 80}")
        header = f"{'Model Name':<32} {'Type':<22} {'Acc':>7} {'F1_bin':>7} {'F1_mac':>7} {'Prec':>7} {'Rec':>7}"
        report_lines.append(header)
        report_lines.append("-" * len(header))

        sorted_res = sorted(results_all, key=lambda r: r[split]["f1_macro"], reverse=True)
        for rank, r in enumerate(sorted_res, 1):
            d = r[split]
            star = " ★" if rank == 1 else ""
            line = (
                f"{r['name']:<32} {r['type']:<22} "
                f"{d['accuracy']:>7.4f} {d['f1_binary']:>7.4f} "
                f"{d['f1_macro']:>7.4f} {d['precision_binary']:>7.4f} "
                f"{d['recall_binary']:>7.4f}{star}"
            )
            report_lines.append(line)

        best = sorted_res[0]
        report_lines.append(f"\n  ★ Best ({split}): {best['name']} — F1 Macro: {best[split]['f1_macro']:.4f}")

    # Confusion Matrices
    report_lines.append(f"\n{'═' * 80}")
    report_lines.append("CONFUSION MATRICES (TEST SET)")
    report_lines.append(f"{'═' * 80}")
    for r in results_all:
        cm = r["test"]["confusion_matrix"]
        report_lines.append(f"\n  {r['name']} ({r['type']}):")
        report_lines.append(f"              Pred:0   Pred:1")
        report_lines.append(f"  True:0     {cm[0][0]:>6}   {cm[0][1]:>6}")
        report_lines.append(f"  True:1     {cm[1][0]:>6}   {cm[1][1]:>6}")

    # Speed & Time
    report_lines.append(f"\n{'═' * 80}")
    report_lines.append("PERFORMANCE & SPEED")
    report_lines.append(f"{'═' * 80}")
    for r in results_all:
        report_lines.append(
            f"  {r['name']:<35} Train: {r['train_time_seconds']:>6.1f}s | "
            f"Infer Speed: {r['samples_per_second']:>7.1f} samples/s"
        )

    report_lines.append("\n" + "=" * 80)
    report_text = "\n".join(report_lines)

    report_path = OUTPUT_DIR / "transformer_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"💾 Report Teks disimpan: {report_path}")
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        safe_text = report_text.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_text)

    # 3. Visualizations
    generate_plots(results_all)


# ──────────────────────────────────────────────────────────────
# Visualizations Function
# ──────────────────────────────────────────────────────────────
def generate_plots(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r["name"] for r in results]
    f1_macros = [r["test"]["f1_macro"] for r in results]
    f1_bins = [r["test"]["f1_binary"] for r in results]
    accuracies = [r["test"]["accuracy"] for r in results]

    x = np.arange(len(names))
    width = 0.25

    rects1 = ax.bar(x - width, f1_macros, width, label='F1 Macro', color='#2b5c8f')
    rects2 = ax.bar(x, f1_bins, width, label='F1 Binary', color='#4682b4')
    rects3 = ax.bar(x + width, accuracies, width, label='Accuracy', color='#87ceeb')

    ax.set_ylabel('Score')
    ax.set_title('Perbandingan Performa Model Transformer (Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Value labels on top of bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "transformer_f1_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"🖼️ Chart disimpan: {chart_path}")

    # Confusion Matrix Grid Plot
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5))
    if n_models == 1:
        axes = [axes]

    for idx, r in enumerate(results):
        cm = np.array(r["test"]["confusion_matrix"])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")

        ax.set_title(r["name"], fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Bully", "Bully"])
        ax.set_yticklabels(["Non-Bully", "Bully"])
        ax.set_xlabel("Predicted")
        if idx == 0:
            ax.set_ylabel("True")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12, fontweight="bold")

    plt.suptitle("Confusion Matrices Model Transformer (Test Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_chart_path = OUTPUT_DIR / "confusion_matrices_transformers.png"
    plt.savefig(cm_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"🖼️ Confusion Matrix Grid disimpan: {cm_chart_path}")


if __name__ == "__main__":
    main()
