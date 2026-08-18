"""
train_xlmr_kfold.py
====================
Stratified 5-Fold Cross-Validation untuk XLM-RoBERTa
pada dataset cyberbullying bilingual (Indonesia + Inggris).

Alur:
  1. Gabung train.csv + val.csv → "development set"
  2. Pisahkan test.csv sebagai held-out test set
  3. StratifiedKFold(K=5) pada development set
  4. Untuk setiap fold: fine-tune dari scratch → evaluasi val & test
  5. Agregasi metrik: mean ± std

Output:
  outputs/kfold/xlmr_kfold_results.json
  outputs/kfold/xlmr_kfold_summary.txt

Cara jalankan:
    python src/models/train_xlmr_kfold.py

Konfigurasi di configs/training_config.yaml
"""

import gc
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
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
logger = logging.getLogger("train_kfold")

# ──────────────────────────────────────────────────────────────
# Paths & Config
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "training_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    """Set seed untuk reproducibility penuh."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Data Validation
# ──────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validasi kolom, missing values, label range, teks kosong, duplikasi."""
    logger.info(f"\n🔍 Validasi data: {name} ({len(df)} baris)")

    required = {"text", "label"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"[{name}] Kolom tidak ditemukan: {missing_cols}")

    na_count = df[["text", "label"]].isna().sum()
    if na_count.any():
        logger.warning(f"  ⚠️ Missing values: {na_count.to_dict()}")
        df = df.dropna(subset=["text", "label"])

    invalid_labels = df[~df["label"].isin([0, 1])]
    if len(invalid_labels) > 0:
        logger.warning(f"  ⚠️ {len(invalid_labels)} baris dengan label invalid")
        df = df[df["label"].isin([0, 1])]

    empty_text = df[df["text"].astype(str).str.strip().str.len() == 0]
    if len(empty_text) > 0:
        logger.warning(f"  ⚠️ {len(empty_text)} baris dengan teks kosong")
        df = df[df["text"].astype(str).str.strip().str.len() > 0]

    dup_count = df.duplicated(subset=["text"], keep="first").sum()
    if dup_count > 0:
        logger.warning(f"  ⚠️ {dup_count} baris duplikat ditemukan")
        df = df.drop_duplicates(subset=["text"], keep="first")

    dist = df["label"].value_counts().to_dict()
    total = len(df)
    for lbl, cnt in sorted(dist.items()):
        pct = cnt / total * 100
        tag = "bully" if lbl == 1 else "non-bully"
        logger.info(f"  Label {lbl} ({tag}): {cnt} ({pct:.1f}%)")

    logger.info(f"  ✅ Valid: {len(df)} baris")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class CyberbullyDataset(Dataset):
    """PyTorch Dataset untuk klasifikasi cyberbullying."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
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
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         label,
        }


# ──────────────────────────────────────────────────────────────
# Extended Metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute metrik lengkap."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)

    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    _, _, f1_wt, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {
        "accuracy":         round(acc, 4),
        "precision_binary": round(p_bin, 4),
        "recall_binary":    round(r_bin, 4),
        "f1_binary":        round(f1_bin, 4),
        "precision_macro":  round(p_mac, 4),
        "recall_macro":     round(r_mac, 4),
        "f1_macro":         round(f1_mac, 4),
        "f1_weighted":      round(f1_wt, 4),
    }


# ──────────────────────────────────────────────────────────────
# WeightedTrainer
# ──────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer dengan weighted loss untuk menangani class imbalance."""

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
# Evaluasi satu split — return dict metrik
# ──────────────────────────────────────────────────────────────
def evaluate_split(trainer, dataset, labels_true, split_name):
    """Evaluasi pada satu split, return dict metrik."""
    logger.info(f"  📊 Evaluasi {split_name} ...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)

    acc = accuracy_score(labels_true, preds)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels_true, preds, average="binary", pos_label=1
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        labels_true, preds, average="macro"
    )
    _, _, f1_wt, _ = precision_recall_fscore_support(
        labels_true, preds, average="weighted"
    )

    cm = confusion_matrix(labels_true, preds)
    report = classification_report(labels_true, preds, target_names=["non-bully", "bully"])

    metrics = {
        "accuracy":         round(acc, 4),
        "precision_binary": round(p_bin, 4),
        "recall_binary":    round(r_bin, 4),
        "f1_binary":        round(f1_bin, 4),
        "precision_macro":  round(p_mac, 4),
        "recall_macro":     round(r_mac, 4),
        "f1_macro":         round(f1_mac, 4),
        "f1_weighted":      round(f1_wt, 4),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(f"    Accuracy: {acc:.4f} | F1 Macro: {f1_mac:.4f} | F1 Binary: {f1_bin:.4f}")
    logger.info(f"    Precision: {p_bin:.4f} | Recall: {r_bin:.4f}")

    return metrics


# ──────────────────────────────────────────────────────────────
# Agregasi metrik across folds
# ──────────────────────────────────────────────────────────────
def aggregate_metrics(fold_metrics_list):
    """Hitung mean ± std dari list of metric dicts."""
    if not fold_metrics_list:
        return {}

    metric_keys = [k for k in fold_metrics_list[0].keys() if k != "confusion_matrix"]
    agg = {}

    for key in metric_keys:
        values = [m[key] for m in fold_metrics_list]
        agg[key] = {
            "mean": round(float(np.mean(values)), 4),
            "std":  round(float(np.std(values)), 4),
            "min":  round(float(np.min(values)), 4),
            "max":  round(float(np.max(values)), 4),
            "per_fold": [round(v, 4) for v in values],
        }

    return agg


# ──────────────────────────────────────────────────────────────
# Generate summary text
# ──────────────────────────────────────────────────────────────
def generate_summary(kfold_results, results_dir):
    """Generate summary teks dan simpan ke file."""
    lines = []
    lines.append("=" * 70)
    lines.append("XLM-ROBERTA K-FOLD CROSS-VALIDATION SUMMARY")
    lines.append(f"Timestamp: {kfold_results['timestamp']}")
    lines.append(f"Model: {kfold_results['model_name']}")
    lines.append(f"K-Folds: {kfold_results['n_splits']}")
    lines.append(f"Seed: {kfold_results['seed']}")
    lines.append("=" * 70)

    for split_name in ["validation", "test"]:
        agg_key = f"{split_name}_aggregate"
        if agg_key not in kfold_results:
            continue

        agg = kfold_results[agg_key]
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {split_name.upper()} SET — Aggregated Results (mean ± std)")
        lines.append(f"{'─' * 70}")

        header = f"  {'Metric':<20} {'Mean':>8} {'± Std':>8} {'Min':>8} {'Max':>8}"
        lines.append(header)
        lines.append("  " + "-" * 56)

        for key, vals in agg.items():
            lines.append(
                f"  {key:<20} {vals['mean']:>8.4f} {vals['std']:>8.4f} "
                f"{vals['min']:>8.4f} {vals['max']:>8.4f}"
            )

        # Per-fold detail
        lines.append(f"\n  Per-fold F1 Macro: {agg['f1_macro']['per_fold']}")

    lines.append("\n" + "=" * 70)

    summary_text = "\n".join(lines)

    summary_path = results_dir / "xlmr_kfold_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"\n📋 Summary disimpan: {summary_path}")

    # Print to console
    try:
        print("\n" + summary_text)
    except UnicodeEncodeError:
        safe_text = summary_text.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_text)

    return summary_text


# ══════════════════════════════════════════════════════════════
# Main K-Fold Pipeline
# ══════════════════════════════════════════════════════════════
def main():
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]
    output_cfg   = cfg["outputs"]
    kfold_cfg    = cfg.get("kfold", {})

    model_name = model_cfg["name"]
    num_labels = model_cfg["num_labels"]
    max_length = model_cfg["max_length"]
    seed       = data_cfg.get("random_seed", 42)
    n_splits   = kfold_cfg.get("n_splits", 5)

    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    results_dir   = PROJECT_ROOT / output_cfg["results_dir"] / "kfold"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Temporary output dir untuk fold checkpoints (akan di-cleanup)
    fold_output_base = PROJECT_ROOT / "models" / "kfold_temp"

    # ── 0. Reproducibility ───────────────────────────────────
    set_all_seeds(seed)

    # ── 1. Load & Validate Data ──────────────────────────────
    logger.info("=" * 60)
    logger.info("XLM-RoBERTa — Stratified K-Fold Cross-Validation")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"K-Folds: {n_splits}")
    logger.info("=" * 60)

    train_df = pd.read_csv(processed_dir / data_cfg["train_file"])
    val_df   = pd.read_csv(processed_dir / data_cfg["val_file"])
    train_df = validate_dataframe(train_df, "Train")
    val_df   = validate_dataframe(val_df, "Validation")

    # Gabung train + val → development set
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    dev_df = dev_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    logger.info(f"\n📦 Development set (train+val): {len(dev_df)} sampel")

    # Load test set (held-out)
    test_path = processed_dir / data_cfg.get("test_file", "test.csv")
    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df = validate_dataframe(test_df, "Test (held-out)")
        logger.info(f"📦 Test set (held-out): {len(test_df)} sampel")

    # ── 2. Tokenizer ─────────────────────────────────────────
    logger.info(f"\nMemuat tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 3. Prepare test dataset (sekali saja) ────────────────
    test_dataset = None
    test_labels = None
    if test_df is not None:
        test_dataset = CyberbullyDataset(
            texts=test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        test_labels = test_df["label"].tolist()

    # ── 4. K-Fold Cross-Validation ───────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    dev_texts  = dev_df["text"].values
    dev_labels = dev_df["label"].values

    val_fold_metrics = []
    test_fold_metrics = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = training_cfg.get("fp16", False) and device == "cuda"

    logger.info(f"\n🚀 Memulai {n_splits}-Fold Cross-Validation ...")
    logger.info(f"   Device: {device}")
    if device == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"   FP16: {use_fp16}")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(dev_texts, dev_labels)):
        logger.info(f"\n{'═' * 60}")
        logger.info(f"📂 FOLD {fold_idx + 1}/{n_splits}")
        logger.info(f"{'═' * 60}")

        fold_train_texts  = dev_texts[train_indices].tolist()
        fold_train_labels = dev_labels[train_indices].tolist()
        fold_val_texts    = dev_texts[val_indices].tolist()
        fold_val_labels   = dev_labels[val_indices].tolist()

        logger.info(f"  Train: {len(fold_train_texts)} sampel")
        logger.info(f"  Val:   {len(fold_val_texts)} sampel")

        # Distribusi label per fold
        train_dist = pd.Series(fold_train_labels).value_counts().to_dict()
        val_dist   = pd.Series(fold_val_labels).value_counts().to_dict()
        logger.info(f"  Train dist: bully={train_dist.get(1,0)}, non-bully={train_dist.get(0,0)}")
        logger.info(f"  Val dist:   bully={val_dist.get(1,0)}, non-bully={val_dist.get(0,0)}")

        # ── Class weights untuk fold ini ─────────────────────
        fold_class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=np.array(fold_train_labels),
        )
        fold_cw_tensor = torch.tensor(fold_class_weights, dtype=torch.float32)

        # ── Datasets ─────────────────────────────────────────
        fold_train_dataset = CyberbullyDataset(
            texts=fold_train_texts,
            labels=fold_train_labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        fold_val_dataset = CyberbullyDataset(
            texts=fold_val_texts,
            labels=fold_val_labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        # ── Model (fresh dari pre-trained setiap fold) ───────
        set_all_seeds(seed)  # Reset seed untuk reproducibility per fold
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={0: "non-bully", 1: "bully"},
            label2id={"non-bully": 0, "bully": 1},
        )

        # ── Training Arguments ───────────────────────────────
        fold_output_dir = fold_output_base / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(fold_output_dir),
            num_train_epochs=training_cfg["num_epochs"],
            per_device_train_batch_size=training_cfg["batch_size_train"],
            per_device_eval_batch_size=training_cfg["batch_size_eval"],
            learning_rate=training_cfg["learning_rate"],
            warmup_ratio=training_cfg["warmup_ratio"],
            weight_decay=training_cfg["weight_decay"],
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            fp16=use_fp16,
            logging_steps=training_cfg.get("logging_steps", 50),
            eval_strategy=training_cfg.get("eval_strategy", "epoch"),
            save_strategy=training_cfg.get("save_strategy", "epoch"),
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            dataloader_num_workers=0,
            logging_dir=str(fold_output_dir / "logs"),
            seed=seed,
        )

        # ── Trainer ──────────────────────────────────────────
        callbacks = []
        patience = training_cfg.get("early_stopping_patience")
        if patience:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        trainer = WeightedTrainer(
            class_weights=fold_cw_tensor,
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            data_collator=data_collator,
        )

        # ── Train ────────────────────────────────────────────
        logger.info(f"  🚀 Training fold {fold_idx + 1} ...")
        train_result = trainer.train()

        logger.info(f"  ✅ Fold {fold_idx + 1} training selesai!")
        logger.info(f"     Steps: {train_result.global_step} | Loss: {train_result.training_loss:.4f}")

        # ── Evaluate validation fold ─────────────────────────
        val_metrics = evaluate_split(
            trainer, fold_val_dataset, fold_val_labels, f"Fold {fold_idx + 1} Val"
        )
        val_fold_metrics.append(val_metrics)

        # ── Evaluate test set ────────────────────────────────
        if test_dataset is not None:
            test_metrics = evaluate_split(
                trainer, test_dataset, test_labels, f"Fold {fold_idx + 1} Test"
            )
            test_fold_metrics.append(test_metrics)

        # ── Cleanup GPU memory ───────────────────────────────
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Cleanup fold checkpoints untuk hemat disk
        import shutil
        if fold_output_dir.exists():
            shutil.rmtree(fold_output_dir, ignore_errors=True)

    # Cleanup temp dir
    import shutil
    if fold_output_base.exists():
        shutil.rmtree(fold_output_base, ignore_errors=True)

    # ── 5. Agregasi ──────────────────────────────────────────
    logger.info(f"\n{'═' * 60}")
    logger.info(f"📊 AGREGASI HASIL {n_splits}-FOLD")
    logger.info(f"{'═' * 60}")

    val_aggregate = aggregate_metrics(val_fold_metrics)
    test_aggregate = aggregate_metrics(test_fold_metrics) if test_fold_metrics else {}

    logger.info(f"\n  Validation (mean ± std):")
    for key, vals in val_aggregate.items():
        logger.info(f"    {key:<20} {vals['mean']:.4f} ± {vals['std']:.4f}")

    if test_aggregate:
        logger.info(f"\n  Test (mean ± std):")
        for key, vals in test_aggregate.items():
            logger.info(f"    {key:<20} {vals['mean']:.4f} ± {vals['std']:.4f}")

    # ── 6. Save Results ──────────────────────────────────────
    kfold_results = {
        "timestamp":    datetime.now().isoformat(),
        "model_name":   model_name,
        "n_splits":     n_splits,
        "seed":         seed,
        "device":       device,
        "config": {
            "epochs":         training_cfg["num_epochs"],
            "learning_rate":  training_cfg["learning_rate"],
            "batch_size":     training_cfg["batch_size_train"],
            "warmup_ratio":   training_cfg["warmup_ratio"],
            "weight_decay":   training_cfg["weight_decay"],
            "fp16":           use_fp16,
            "max_grad_norm":  training_cfg.get("max_grad_norm", 1.0),
            "early_stopping": patience,
        },
        "data": {
            "development_samples": len(dev_df),
            "test_samples":        len(test_df) if test_df is not None else 0,
        },
        "validation_per_fold":  val_fold_metrics,
        "test_per_fold":        test_fold_metrics,
        "validation_aggregate": val_aggregate,
        "test_aggregate":       test_aggregate,
    }

    results_path = results_dir / "xlmr_kfold_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(kfold_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n💾 Hasil K-Fold disimpan: {results_path}")

    # ── 7. Generate Summary ──────────────────────────────────
    generate_summary(kfold_results, results_dir)

    # ── Final ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ XLM-ROBERTA K-FOLD CROSS-VALIDATION SELESAI!")
    logger.info("=" * 60)
    logger.info(f"   Folds:              {n_splits}")
    logger.info(f"   Val F1 Macro:       {val_aggregate['f1_macro']['mean']:.4f} ± {val_aggregate['f1_macro']['std']:.4f}")
    if test_aggregate:
        logger.info(f"   Test F1 Macro:      {test_aggregate['f1_macro']['mean']:.4f} ± {test_aggregate['f1_macro']['std']:.4f}")
    logger.info(f"   Results:            {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
