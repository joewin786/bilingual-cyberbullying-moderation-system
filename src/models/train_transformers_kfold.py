"""
train_transformers_kfold.py
============================
Script Stratified K-Fold Cross-Validation (K=5) untuk membandingkan
model berbasis Transformer (XLM-RoBERTa, IndoBERT, mBERT, IndoLEM)
pada dataset cyberbullying bilingual.

Alur:
  1. Gabung train.csv + val.csv → "development set"
  2. Hold-out test.csv sebagai test set independen
  3. StratifiedKFold(K=5) pada development set
  4. Untuk setiap fold & tiap model: fine-tune dari scratch → evaluasi val & test
  5. Hitung rata-rata dan standar deviasi (mean ± std) F1 Macro & Accuracy.

Cara jalankan:
    python src/models/train_transformers_kfold.py

Output:
    outputs/kfold/
        transformers_kfold_results.json
        transformers_kfold_summary.txt
        transformers_kfold_f1_comparison.png
"""

import gc
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
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
logger = logging.getLogger("transformers_kfold")

# ──────────────────────────────────────────────────────────────
# Paths & Config
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kfold"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "kfold_transformers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
RANDOM_SEED = 42
N_SPLITS = 5
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16_ENABLED = torch.cuda.is_available()

# Models to evaluate in K-Fold
MODELS_TO_COMPARE = [
    {
        "key": "xlm-roberta",
        "name": "XLM-RoBERTa Base",
        "hf_id": "xlm-roberta-base",
        "type": "Multilingual",
    },
    {
        "key": "indobert-p1",
        "name": "IndoBERT Base (IndoBenchmark)",
        "hf_id": "indobenchmark/indobert-base-p1",
        "type": "Indonesian Monolingual",
    },
    {
        "key": "mbert-cased",
        "name": "mBERT Base (Multilingual)",
        "hf_id": "bert-base-multilingual-cased",
        "type": "Multilingual",
    },
    {
        "key": "indobert-indolem",
        "name": "IndoBERT Uncased (IndoLEM)",
        "hf_id": "indolem/indobert-base-uncased",
        "type": "Indonesian Monolingual",
    },
]


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🌱 Global seed di-set ke: {seed}")


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
        "accuracy": round(float(acc), 4),
        "precision_binary": round(float(p_bin), 4),
        "recall_binary": round(float(r_bin), 4),
        "f1_binary": round(float(f1_bin), 4),
        "precision_macro": round(float(p_mac), 4),
        "recall_macro": round(float(r_mac), 4),
        "f1_macro": round(float(f1_mac), 4),
        "f1_weighted": round(float(f1_wt), 4),
    }


def evaluate_split(trainer, dataset, labels_true):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    acc = accuracy_score(labels_true, preds)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels_true, preds, average="binary", pos_label=1, zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        labels_true, preds, average="macro", zero_division=0
    )
    _, _, f1_wt, _ = precision_recall_fscore_support(
        labels_true, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(labels_true, preds)

    return {
        "accuracy": float(acc),
        "precision_binary": float(p_bin),
        "recall_binary": float(r_bin),
        "f1_binary": float(f1_bin),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wt),
        "confusion_matrix": cm.tolist(),
    }


def aggregate_fold_metrics(fold_results, split_key="val"):
    metrics_keys = [
        "accuracy",
        "precision_binary",
        "recall_binary",
        "f1_binary",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]
    summary = {}
    for m in metrics_keys:
        vals = [f[split_key][m] for f in fold_results]
        summary[m] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "values": [round(float(v), 4) for v in vals],
        }
    return summary


def find_saved_fold_checkpoint(fold_output_dir):
    best_dir = fold_output_dir / "best_model"
    if best_dir.exists() and (best_dir / "config.json").exists():
        return best_dir
    ckpt_dir = fold_output_dir / "checkpoints"
    if ckpt_dir.exists():
        subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
        if subdirs:
            return sorted(subdirs, key=lambda x: x.stat().st_mtime)[-1]
    return None


def main():
    set_all_seeds(RANDOM_SEED)

    logger.info("=" * 70)
    logger.info(f"🚀 STRATIFIED {N_SPLITS}-FOLD CROSS-VALIDATION TRANSFORMER BENCHMARK")
    logger.info("=" * 70)

    # 1. Load Data
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    dev_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    logger.info(f"Development Set (Train+Val): {len(dev_df)} samples")
    logger.info(f"Held-out Test Set: {len(test_df)} samples")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    all_model_results = []

    for model_info in MODELS_TO_COMPARE:
        key = model_info["key"]
        name = model_info["name"]
        hf_id = model_info["hf_id"]
        model_type = model_info["type"]

        logger.info("\n" + "═" * 70)
        logger.info(f"MODEL: {name} [{hf_id}] ({model_type})")
        logger.info("═" * 70)

        fold_results = []
        total_train_time = 0.0

        for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df["label"]), 1):
            logger.info(f"\n--- Model {key} | Fold {fold}/{N_SPLITS} ---")

            train_fold_df = dev_df.iloc[train_idx].reset_index(drop=True)
            val_fold_df = dev_df.iloc[val_idx].reset_index(drop=True)

            classes = np.unique(train_fold_df["label"])
            weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=train_fold_df["label"].values
            )
            class_weights = torch.tensor(weights, dtype=torch.float)

            fold_output_dir = MODEL_SAVE_DIR / key / f"fold_{fold}"
            saved_ckpt = find_saved_fold_checkpoint(fold_output_dir)

            if saved_ckpt:
                logger.info(f"  ✅ Saved checkpoint fold {fold} ditemukan di: {saved_ckpt}. Memuat model...")
                tokenizer = AutoTokenizer.from_pretrained(str(saved_ckpt))
                model = AutoModelForSequenceClassification.from_pretrained(str(saved_ckpt))
                fold_time = 0.0
            else:
                logger.info(f"  📥 Training fold {fold} dari HuggingFace: {hf_id}...")
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=2)

                train_ds = CyberbullyDataset(train_fold_df["text"], train_fold_df["label"], tokenizer, MAX_LENGTH)
                val_ds = CyberbullyDataset(val_fold_df["text"], val_fold_df["label"], tokenizer, MAX_LENGTH)

                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                training_args = TrainingArguments(
                    output_dir=str(fold_output_dir / "checkpoints"),
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
                    logging_steps=100,
                    report_to="none",
                    seed=RANDOM_SEED + fold,
                )

                trainer = WeightedTrainer(
                    class_weights=class_weights,
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                )

                t0 = time.time()
                trainer.train()
                fold_time = round(time.time() - t0, 2)

                # Save best model to best_model dir
                best_dir = fold_output_dir / "best_model"
                best_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_model(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
                logger.info(f"  💾 Best model fold {fold} disimpan di: {best_dir}")

            total_train_time += fold_time

            val_ds = CyberbullyDataset(val_fold_df["text"], val_fold_df["label"], tokenizer, MAX_LENGTH)
            test_ds = CyberbullyDataset(test_df["text"], test_df["label"], tokenizer, MAX_LENGTH)
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

            val_metrics = evaluate_split(eval_trainer, val_ds, val_fold_df["label"].values)
            test_metrics = evaluate_split(eval_trainer, test_ds, test_df["label"].values)

            logger.info(
                f"  Fold {fold} selesai ({fold_time}s) -> "
                f"Val F1-Macro: {val_metrics['f1_macro']:.4f} | "
                f"Test F1-Macro: {test_metrics['f1_macro']:.4f}"
            )

            fold_results.append({
                "fold": fold,
                "train_time_seconds": fold_time,
                "val": val_metrics,
                "test": test_metrics,
            })

            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_agg = aggregate_fold_metrics(fold_results, "val")
        test_agg = aggregate_fold_metrics(fold_results, "test")

        model_entry = {
            "key": key,
            "name": name,
            "hf_id": hf_id,
            "type": model_type,
            "total_train_time_seconds": round(total_train_time, 2),
            "folds": fold_results,
            "aggregated_validation": val_agg,
            "aggregated_test": test_agg,
        }
        all_model_results.append(model_entry)

    # ──────────────────────────────────────────────────────────
    # SAVE & REPORT
    # ──────────────────────────────────────────────────────────
    json_path = OUTPUT_DIR / "transformers_kfold_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, indent=2)
    logger.info(f"💾 JSON disimpan: {json_path}")

    # Generate Summary Report
    lines = []
    lines.append("=" * 80)
    lines.append(f"STRATIFIED {N_SPLITS}-FOLD CROSS-VALIDATION TRANSFORMER COMPARISON")
    lines.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    for split_key, split_title in [("aggregated_validation", "VALIDATION FOLDS"), ("aggregated_test", "HELD-OUT TEST SET")]:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"  {split_title} RESULTS (mean ± std)")
        lines.append(f"{'─' * 80}")
        header = f"{'Model Name':<32} {'Type':<22} {'Acc (mean±std)':<18} {'F1-Macro (mean±std)':<18}"
        lines.append(header)
        lines.append("-" * len(header))

        sorted_models = sorted(all_model_results, key=lambda m: m[split_key]["f1_macro"]["mean"], reverse=True)
        for rank, m in enumerate(sorted_models, 1):
            acc_m = m[split_key]["accuracy"]
            f1m_m = m[split_key]["f1_macro"]
            star = " ★" if rank == 1 else ""
            line = (
                f"{m['name']:<32} {m['type']:<22} "
                f"{acc_m['mean']:.4f} ± {acc_m['std']:.4f}   "
                f"{f1m_m['mean']:.4f} ± {f1m_m['std']:.4f}{star}"
            )
            lines.append(line)

        best = sorted_models[0]
        lines.append(f"\n  ★ Best ({split_title}): {best['name']} — F1 Macro: {best[split_key]['f1_macro']['mean']:.4f} ± {best[split_key]['f1_macro']['std']:.4f}")

    lines.append("\n" + "=" * 80)
    report_text = "\n".join(lines)

    summary_path = OUTPUT_DIR / "transformers_kfold_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"💾 Summary Report disimpan: {summary_path}")

    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        safe_text = report_text.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_text)

    generate_kfold_plots(all_model_results)


def generate_kfold_plots(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r["name"] for r in results]
    val_means = [r["aggregated_validation"]["f1_macro"]["mean"] for r in results]
    val_stds = [r["aggregated_validation"]["f1_macro"]["std"] for r in results]
    test_means = [r["aggregated_test"]["f1_macro"]["mean"] for r in results]
    test_stds = [r["aggregated_test"]["f1_macro"]["std"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    rects1 = ax.bar(x - width/2, val_means, width, yerr=val_stds, label='Val Folds (mean ± std)', capsize=5, color='#2b5c8f')
    rects2 = ax.bar(x + width/2, test_means, width, yerr=test_stds, label='Test Set (mean ± std)', capsize=5, color='#4682b4')

    ax.set_ylabel('F1 Macro Score')
    ax.set_title(f'Stratified 5-Fold Cross-Validation Transformer Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0.7, 0.9)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "transformers_kfold_f1_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logger.info(f"🖼️ K-Fold Chart disimpan: {chart_path}")


if __name__ == "__main__":
    main()
