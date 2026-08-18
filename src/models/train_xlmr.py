"""
train_xlmr.py
=============
Standalone training script untuk fine-tuning XLM-RoBERTa
pada dataset cyberbullying bilingual (Indonesia + Inggris).

Fitur:
  - Metrik lengkap (binary, macro, weighted F1)
  - Class-weighted loss untuk menangani class imbalance
  - Dynamic padding (DataCollatorWithPadding)
  - Reproducibility (seed global)
  - Validasi data sebelum training
  - Threshold tuning pada validation set
  - Error analysis (false positive / false negative)
  - Evaluation report untuk val dan test set

Cara jalankan:
    python src/models/train_xlmr.py

Konfigurasi di configs/training_config.yaml
"""

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
logger = logging.getLogger("train")

# ──────────────────────────────────────────────────────────────
# Paths & Config
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "training_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Reproducibility — set semua seed global
# ──────────────────────────────────────────────────────────────
def set_all_seeds(seed: int):
    """Set seed untuk reproducibility penuh."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    # Deterministic backends (sedikit lebih lambat, tapi reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🌱 Seed di-set ke: {seed}")


# ──────────────────────────────────────────────────────────────
# Data Validation — cek kualitas data sebelum training
# ──────────────────────────────────────────────────────────────
def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validasi kolom, missing values, label range, teks kosong, duplikasi."""
    logger.info(f"\n🔍 Validasi data: {name} ({len(df)} baris)")

    # Cek kolom wajib
    required = {"text", "label"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"[{name}] Kolom tidak ditemukan: {missing_cols}")

    # Missing values
    na_count = df[["text", "label"]].isna().sum()
    if na_count.any():
        logger.warning(f"  ⚠️ Missing values: {na_count.to_dict()}")
        df = df.dropna(subset=["text", "label"])
        logger.info(f"  → Setelah drop NA: {len(df)} baris")

    # Label di luar 0/1
    invalid_labels = df[~df["label"].isin([0, 1])]
    if len(invalid_labels) > 0:
        logger.warning(f"  ⚠️ {len(invalid_labels)} baris dengan label invalid (bukan 0/1)")
        df = df[df["label"].isin([0, 1])]

    # Teks kosong
    empty_text = df[df["text"].astype(str).str.strip().str.len() == 0]
    if len(empty_text) > 0:
        logger.warning(f"  ⚠️ {len(empty_text)} baris dengan teks kosong")
        df = df[df["text"].astype(str).str.strip().str.len() > 0]

    # Duplikasi
    dup_count = df.duplicated(subset=["text"], keep="first").sum()
    if dup_count > 0:
        logger.warning(f"  ⚠️ {dup_count} baris duplikat ditemukan")
        df = df.drop_duplicates(subset=["text"], keep="first")

    # Distribusi label
    dist = df["label"].value_counts().to_dict()
    total = len(df)
    for lbl, cnt in sorted(dist.items()):
        pct = cnt / total * 100
        tag = "bully" if lbl == 1 else "non-bully"
        logger.info(f"  Label {lbl} ({tag}): {cnt} ({pct:.1f}%)")

    logger.info(f"  ✅ Valid: {len(df)} baris")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Dataset — dynamic padding (tokenizer hanya truncation)
# ──────────────────────────────────────────────────────────────
class CyberbullyDataset(Dataset):
    """PyTorch Dataset untuk klasifikasi cyberbullying.

    Tokenizer hanya melakukan truncation di sini.
    Padding dilakukan secara dinamis oleh DataCollatorWithPadding
    sehingga batch lebih efisien (tidak pad ke max_length setiap sample).
    """

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

        # Dynamic padding: hanya truncation, TANPA padding di sini
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            # Tidak ada padding="max_length" — collator yang handle
        )

        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         label,
        }


# ──────────────────────────────────────────────────────────────
# Extended Metrics — binary + macro + weighted
# ──────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Compute metrik lengkap: accuracy, precision/recall/f1 (binary, macro, weighted)."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)

    # Binary (pos_label=1 → bully)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )
    # Macro (rata-rata tanpa bobot antar kelas)
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    # Weighted (bobot sesuai jumlah sampel per kelas)
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
# WeightedTrainer — class-weighted CrossEntropy loss
# ──────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer dengan weighted loss untuk menangani class imbalance.

    Override compute_loss agar menggunakan CrossEntropyLoss(weight=class_weights).
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Pindahkan weight ke device yang sama dengan logits
        weight = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ──────────────────────────────────────────────────────────────
# Evaluation Report — val + test, classification report + CM
# ──────────────────────────────────────────────────────────────
def generate_eval_report(
    trainer: Trainer,
    dataset: Dataset,
    labels_true: list,
    split_name: str,
    results_dir: Path,
    threshold: float = None,
) -> dict:
    """Generate classification report + confusion matrix untuk sebuah split."""
    logger.info(f"\n📊 Evaluasi pada {split_name} set (Threshold: {threshold if threshold is not None else 0.50}) ...")

    # Bersihkan GPU cache sebelum evaluasi untuk mencegah OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Gunakan predict() saja (sudah termasuk metrics) — lebih hemat memori
    # daripada memanggil evaluate() + predict() terpisah
    predictions = trainer.predict(dataset)
    logits = predictions.predictions

    if threshold is not None:
        probs = softmax(logits, axis=-1)
        bully_probs = probs[:, 1]
        preds = (bully_probs >= threshold).astype(int)

        # Hitung metrik ulang untuk threshold ini
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

        eval_results = {
            "test_loss": predictions.metrics.get("test_loss", predictions.metrics.get("eval_loss", 0.0)),
            "test_accuracy": acc,
            "test_precision_binary": p_bin,
            "test_recall_binary": r_bin,
            "test_f1_binary": f1_bin,
            "test_precision_macro": p_mac,
            "test_recall_macro": r_mac,
            "test_f1_macro": f1_mac,
            "test_f1_weighted": f1_wt,
            "test_runtime": predictions.metrics.get("test_runtime", 0.0),
            "test_samples_per_second": predictions.metrics.get("test_samples_per_second", 0.0),
            "test_steps_per_second": predictions.metrics.get("test_steps_per_second", 0.0),
        }
    else:
        eval_results = predictions.metrics
        preds = np.argmax(logits, axis=-1)

    logger.info(f"{split_name} results:")
    for k, v in eval_results.items():
        logger.info(f"   {k}: {v}")

    # Classification report
    report = classification_report(
        labels_true, preds, target_names=["non-bully", "bully"]
    )
    logger.info(f"\n{split_name} Classification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(labels_true, preds)
    logger.info(f"{split_name} Confusion Matrix:\n{cm}")

    # Save to file
    report_path = results_dir / f"evaluation_report_{split_name.lower()}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"EVALUATION REPORT — {split_name.upper()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"CLASSIFICATION REPORT\n{'-'*40}\n")
        f.write(report + "\n\n")
        f.write(f"CONFUSION MATRIX\n{'-'*40}\n")
        f.write(f"            Pred:0   Pred:1\n")
        f.write(f"True:0     {cm[0][0]:>6}   {cm[0][1]:>6}\n")
        f.write(f"True:1     {cm[1][0]:>6}   {cm[1][1]:>6}\n")
    logger.info(f"📋 Report disimpan: {report_path}")

    return {
        "metrics": {k: round(v, 4) if isinstance(v, float) else v
                    for k, v in eval_results.items()},
        "predictions": preds,
        "logits": logits,
    }


# ──────────────────────────────────────────────────────────────
# Threshold Tuning — cari threshold optimal berdasarkan macro F1
# ──────────────────────────────────────────────────────────────
def tune_threshold(
    logits: np.ndarray,
    labels_true: list,
    thresholds: list = None,
) -> tuple:
    """Uji beberapa threshold pada probabilitas kelas bully.

    Returns:
        (best_threshold, best_f1_macro, threshold_results)
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    probs = softmax(logits, axis=-1)
    bully_probs = probs[:, 1]  # Probabilitas kelas bully

    results = []
    best_f1, best_thr = -1, 0.5

    logger.info("\n🎯 Threshold Tuning (berdasarkan macro F1):")
    logger.info(f"   {'Threshold':>10} | {'Acc':>6} | {'F1_mac':>7} | {'F1_bin':>7} | {'Prec':>6} | {'Rec':>6}")
    logger.info(f"   {'-'*10}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}")

    for thr in thresholds:
        preds = (bully_probs >= thr).astype(int)
        acc = accuracy_score(labels_true, preds)
        p, r, f1_bin, _ = precision_recall_fscore_support(
            labels_true, preds, average="binary", pos_label=1, zero_division=0
        )
        _, _, f1_mac, _ = precision_recall_fscore_support(
            labels_true, preds, average="macro", zero_division=0
        )

        marker = ""
        if f1_mac > best_f1:
            best_f1, best_thr = f1_mac, thr
            marker = " ← best"

        logger.info(
            f"   {thr:>10.2f} | {acc:>6.4f} | {f1_mac:>7.4f} | {f1_bin:>7.4f} | {p:>6.4f} | {r:>6.4f}{marker}"
        )
        results.append({
            "threshold": thr, "accuracy": round(acc, 4),
            "f1_macro": round(f1_mac, 4), "f1_binary": round(f1_bin, 4),
            "precision": round(p, 4), "recall": round(r, 4),
        })

    logger.info(f"\n   ✅ Best threshold: {best_thr:.2f} (macro F1={best_f1:.4f})")
    return best_thr, best_f1, results


# ──────────────────────────────────────────────────────────────
# Error Analysis — simpan contoh FP dan FN
# ──────────────────────────────────────────────────────────────
def error_analysis(
    texts: list,
    labels_true: list,
    preds: np.ndarray,
    logits: np.ndarray,
    results_dir: Path,
    split_name: str = "test",
    max_examples: int = 20,
):
    """Simpan contoh false positive dan false negative ke file."""
    probs = softmax(logits, axis=-1)
    labels_true = np.array(labels_true)

    # False Positive: model prediksi bully, tapi sebenarnya non-bully
    fp_mask = (preds == 1) & (labels_true == 0)
    # False Negative: model prediksi non-bully, tapi sebenarnya bully
    fn_mask = (preds == 0) & (labels_true == 1)

    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]

    errors = {
        "split": split_name,
        "total_samples": len(texts),
        "total_fp": int(fp_mask.sum()),
        "total_fn": int(fn_mask.sum()),
        "false_positives": [],
        "false_negatives": [],
    }

    # Urutkan berdasarkan confidence tertinggi (paling 'yakin' tapi salah)
    fp_sorted = sorted(fp_indices, key=lambda i: probs[i][1], reverse=True)
    for i in fp_sorted[:max_examples]:
        errors["false_positives"].append({
            "text": texts[i],
            "true_label": "non-bully",
            "pred_label": "bully",
            "confidence_bully": round(float(probs[i][1]), 4),
        })

    fn_sorted = sorted(fn_indices, key=lambda i: probs[i][0], reverse=True)
    for i in fn_sorted[:max_examples]:
        errors["false_negatives"].append({
            "text": texts[i],
            "true_label": "bully",
            "pred_label": "non-bully",
            "confidence_bully": round(float(probs[i][1]), 4),
        })

    # Simpan ke file
    error_path = results_dir / f"error_analysis_{split_name.lower()}.json"
    with open(error_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    logger.info(f"\n🔎 Error Analysis ({split_name}):")
    logger.info(f"   False Positives: {errors['total_fp']}")
    logger.info(f"   False Negatives: {errors['total_fn']}")
    logger.info(f"   📋 Detail disimpan: {error_path}")

    return errors


# ══════════════════════════════════════════════════════════════
# Main Training Pipeline
# ══════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path ke checkpoint untuk melanjutkan training (misal: models/xlmr_cyberbully/checkpoint-1008)"
    )
    args, _ = parser.parse_known_args()
    resume_checkpoint = args.resume_from_checkpoint
    if resume_checkpoint:
        logger.info(f"\n🔄 Melanjutkan training dari checkpoint: {resume_checkpoint}")

    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    data_cfg     = cfg["data"]
    output_cfg   = cfg["outputs"]

    model_name = model_cfg["name"]
    num_labels = model_cfg["num_labels"]
    max_length = model_cfg["max_length"]
    seed       = data_cfg.get("random_seed", 42)

    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    output_dir    = PROJECT_ROOT / training_cfg["output_dir"]
    results_dir   = PROJECT_ROOT / output_cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Reproducibility ───────────────────────────────────
    set_all_seeds(seed)

    # ── 1. Load & Validate Data ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Cyberbullying Detection — Training Pipeline")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    train_df = pd.read_csv(processed_dir / data_cfg["train_file"])
    val_df   = pd.read_csv(processed_dir / data_cfg["val_file"])

    train_df = validate_dataframe(train_df, "Train")
    val_df   = validate_dataframe(val_df, "Validation")

    # Load test jika ada
    test_path = processed_dir / data_cfg.get("test_file", "test.csv")
    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        test_df = validate_dataframe(test_df, "Test")

    # ── 2. Class Weights ─────────────────────────────────────
    train_labels = train_df["label"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"\n⚖️ Class weights (balanced): {dict(enumerate(np.round(class_weights, 4)))}")

    # ── 3. Tokenizer + Datasets ──────────────────────────────
    logger.info(f"\nMemuat tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = CyberbullyDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = CyberbullyDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Dynamic padding collator — pad ke panjang terpanjang dalam batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 4. Model ─────────────────────────────────────────────
    logger.info(f"Memuat model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={0: "non-bully", 1: "bully"},
        label2id={"non-bully": 0, "bully": 1},
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 5. Training Arguments ────────────────────────────────
    use_fp16 = training_cfg.get("fp16", False) and device == "cuda"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["batch_size_train"],
        per_device_eval_batch_size=training_cfg["batch_size_eval"],
        learning_rate=training_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),  # Gradient clipping
        fp16=use_fp16,
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",          # Best model berdasarkan macro F1
        greater_is_better=True,
        save_total_limit=training_cfg.get("save_total_limit", 1),  # ⚡ Baca dari config (default 1 = hemat disk)
        report_to="none",
        dataloader_num_workers=0,
        logging_dir=str(results_dir / "logs"),
        seed=seed,
    )

    # ── 6. WeightedTrainer ───────────────────────────────────
    callbacks = []
    patience = training_cfg.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        data_collator=data_collator,  # Dynamic padding
    )

    # ── 7. Train ─────────────────────────────────────────────
    logger.info("\n🚀 Memulai training ...")
    logger.info(f"   Model:        {model_name}")
    logger.info(f"   Epochs:       {training_cfg['num_epochs']}")
    logger.info(f"   Batch size:   {training_cfg['batch_size_train']}")
    logger.info(f"   LR:           {training_cfg['learning_rate']}")
    logger.info(f"   FP16:         {use_fp16}")
    logger.info(f"   Seed:         {seed}")
    logger.info(f"   Metric:       f1_macro")

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    logger.info(f"\n✅ Training selesai!")
    logger.info(f"   Total steps:  {train_result.global_step}")
    logger.info(f"   Train loss:   {train_result.training_loss:.4f}")

    # Cari epoch terbaik dari log history
    best_metric_val = -1
    best_epoch = -1
    for entry in trainer.state.log_history:
        if "eval_f1_macro" in entry:
            if entry["eval_f1_macro"] > best_metric_val:
                best_metric_val = entry["eval_f1_macro"]
                best_epoch = entry.get("epoch", -1)
    logger.info(f"   Best epoch:   {best_epoch} (f1_macro={best_metric_val:.4f})")

    # ── 8. Evaluation — Validation Set ───────────────────────
    val_report = generate_eval_report(
        trainer, val_dataset,
        val_df["label"].tolist(), "Validation", results_dir,
    )

    # ── 9. Save Best Model ───────────────────────────────────
    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info(f"\n💾 Model terbaik disimpan ke: {best_model_dir}")

    # ── 10. Threshold Tuning — pada Validation Set ───────────
    best_threshold, best_thr_f1, thr_results = tune_threshold(
        val_report["logits"], val_df["label"].tolist()
    )

    # ── 11. Error Analysis — Validation Set ──────────────────
    val_errors = error_analysis(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        val_report["predictions"],
        val_report["logits"],
        results_dir,
        split_name="validation",
    )

    # ── 12. Evaluation — Test Set ────────────────────────────
    test_report = None
    test_errors = None
    if test_df is not None:
        # Bersihkan GPU memory sebelum test evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        test_dataset = CyberbullyDataset(
            texts=test_df["text"].tolist(),
            labels=test_df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        test_report = generate_eval_report(
            trainer, test_dataset,
            test_df["label"].tolist(), "Test", results_dir,
            threshold=best_threshold,
        )

        # Threshold tuning pada test set (dengan threshold terbaik dari val)
        test_thr, test_thr_f1, test_thr_results = tune_threshold(
            test_report["logits"], test_df["label"].tolist()
        )

        # Error analysis pada test set
        test_errors = error_analysis(
            test_df["text"].tolist(),
            test_df["label"].tolist(),
            test_report["predictions"],
            test_report["logits"],
            results_dir,
            split_name="test",
        )

    # ── 13. Save Complete Experiment Results ──────────────────
    experiment_results = {
        "timestamp":      datetime.now().isoformat(),
        "seed":           seed,
        "model_name":     model_name,
        "num_labels":     num_labels,
        "max_length":     max_length,
        "device":         device,
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
            "train_samples":  len(train_df),
            "val_samples":    len(val_df),
            "test_samples":   len(test_df) if test_df is not None else 0,
            "class_distribution": {
                "train": train_df["label"].value_counts().to_dict(),
                "val":   val_df["label"].value_counts().to_dict(),
                "test":  test_df["label"].value_counts().to_dict() if test_df is not None else {},
            },
        },
        "class_weights":  {str(i): round(w, 4) for i, w in enumerate(class_weights)},
        "training": {
            "total_steps":    train_result.global_step,
            "train_loss":     round(train_result.training_loss, 4),
            "best_epoch":     best_epoch,
            "best_f1_macro":  round(best_metric_val, 4),
        },
        "validation_metrics": val_report["metrics"],
        "test_metrics":       test_report["metrics"] if test_report else None,
        "threshold_tuning": {
            "best_threshold":     best_threshold,
            "best_val_f1_macro":  round(best_thr_f1, 4),
            "all_thresholds":     thr_results,
        },
        "error_summary": {
            "validation": {
                "false_positives": val_errors["total_fp"],
                "false_negatives": val_errors["total_fn"],
            },
            "test": {
                "false_positives": test_errors["total_fp"],
                "false_negatives": test_errors["total_fn"],
            } if test_errors else None,
        },
        "best_model_dir": str(best_model_dir),
    }

    results_path = results_dir / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n📋 Hasil eksperimen lengkap disimpan: {results_path}")

    # ── Summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING PIPELINE SELESAI!")
    logger.info("=" * 60)
    logger.info(f"   Model:          {model_name}")
    logger.info(f"   Best epoch:     {best_epoch}")
    logger.info(f"   Val f1_macro:   {best_metric_val:.4f}")
    if test_report:
        test_f1m = test_report["metrics"].get("eval_f1_macro", "N/A")
        logger.info(f"   Test f1_macro:  {test_f1m}")
    logger.info(f"   Best threshold: {best_threshold:.2f}")
    logger.info(f"   Model saved:    {best_model_dir}")
    logger.info(f"   Results saved:  {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
