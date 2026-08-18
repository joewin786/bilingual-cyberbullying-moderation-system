"""
evaluate_label_quality_kfold.py
===============================
Script untuk mengukur kualitas label (label quality assessment) dan mendeteksi
derau label (label noise / mislabeled instances) pada merged dataset (Development Set: 24.951 sampel)
dengan memanfaatkan Out-Of-Fold (OOF) predictions dari 5-Fold Cross-Validation XLM-RoBERTa.

Alur:
1. Load train.csv & val.csv -> dev_df (24.951 sampel)
2. StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
3. Load checkpoint model XLM-RoBERTa fold 1 s.d. 5
4. Lakukan inferensi OOF pada validation fold masing-masing
5. Gabung probabilitas OOF per sampel
6. Hitung metrik Kualitas Label (Cross-Entropy Loss, Disagreement Rate, High-Confidence Noise)
7. Simpan daftar sampel berisiko ke outputs/suspected_mislabeled_samples.csv
8. Buat laporan ringkas ke outputs/label_quality_report.md
"""

import json
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("label_quality_eval")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "kfold_transformers" / "xlm-roberta"

RANDOM_SEED = 42
N_SPLITS = 5
MAX_LENGTH = 128
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = [str(t) for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

def find_fold_checkpoint(fold_num):
    fold_dir = MODEL_SAVE_DIR / f"fold_{fold_num}"
    best_dir = fold_dir / "best_model"
    if best_dir.exists() and (best_dir / "config.json").exists():
        return best_dir
    ckpt_dir = fold_dir / "checkpoints"
    if ckpt_dir.exists():
        subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
        if subdirs:
            return sorted(subdirs, key=lambda x: x.stat().st_mtime)[-1]
    return None

def run_oof_inference():
    logger.info("=" * 70)
    logger.info("🚀 PENGUKURAN KUALITAS LABEL DENGAN OUT-OF-FOLD (OOF) 5-FOLD CV")
    logger.info(f"Menggunakan Device: {DEVICE}")
    logger.info("=" * 70)

    # 1. Load Data
    train_csv = PROCESSED_DIR / "train.csv"
    val_csv = PROCESSED_DIR / "val.csv"
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("File train.csv atau val.csv tidak ditemukan di data/processed/")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Tambahkan tag split asli jika belum ada
    train_df["orig_split"] = "train"
    val_df["orig_split"] = "val"

    dev_df = pd.concat([train_df, val_df], ignore_index=True).reset_index(drop=True)
    logger.info(f"Total Development Set: {len(dev_df)} sampel")

    # Inisialisasi kolom OOF
    dev_df["oof_prob_0"] = 0.0
    dev_df["oof_prob_1"] = 0.0
    dev_df["oof_pred"] = -1
    dev_df["oof_fold"] = -1

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df["label"]), 1):
        logger.info(f"\n--- Memproses Fold {fold}/{N_SPLITS} ({len(val_idx)} sampel validasi OOF) ---")
        
        ckpt_path = find_fold_checkpoint(fold)
        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint untuk fold {fold} tidak ditemukan di {MODEL_SAVE_DIR}")

        logger.info(f"  Memuat model dari: {ckpt_path.name}")
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(ckpt_path)).to(DEVICE)
        model.eval()

        val_fold_df = dev_df.iloc[val_idx]
        val_ds = InferenceDataset(val_fold_df["text"], tokenizer, MAX_LENGTH)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                probs_list.append(probs)

        probs_all = np.vstack(probs_list)
        
        dev_df.loc[val_idx, "oof_prob_0"] = probs_all[:, 0]
        dev_df.loc[val_idx, "oof_prob_1"] = probs_all[:, 1]
        dev_df.loc[val_idx, "oof_pred"] = np.argmax(probs_all, axis=1)
        dev_df.loc[val_idx, "oof_fold"] = fold

    # 2. Hitung Metrik Kualitas Label & Noise
    # A. Cross-Entropy Loss per sampel
    eps = 1e-15
    probs_true_class = np.where(
        dev_df["label"] == 1,
        np.clip(dev_df["oof_prob_1"], eps, 1 - eps),
        np.clip(dev_df["oof_prob_0"], eps, 1 - eps)
    )
    dev_df["ce_loss"] = -np.log(probs_true_class)

    # B. Disagreement / Mismatch
    dev_df["is_disagreement"] = (dev_df["label"] != dev_df["oof_pred"]).astype(int)

    # C. High-confidence label noise
    # False Positive noise: label = 0 tapi OOF prob_1 >= 0.80
    # False Negative noise: label = 1 tapi OOF prob_1 <= 0.20
    dev_df["noise_type"] = "Clean"
    dev_df.loc[(dev_df["label"] == 0) & (dev_df["oof_prob_1"] >= 0.80), "noise_type"] = "Suspected_False_Positive"
    dev_df.loc[(dev_df["label"] == 1) & (dev_df["oof_prob_1"] <= 0.20), "noise_type"] = "Suspected_False_Negative"

    # 3. Statistik Keseluruhan
    total_samples = len(dev_df)
    total_disagreements = dev_df["is_disagreement"].sum()
    disagreement_rate = (total_disagreements / total_samples) * 100

    n_suspected_fp = (dev_df["noise_type"] == "Suspected_False_Positive").sum()
    n_suspected_fn = (dev_df["noise_type"] == "Suspected_False_Negative").sum()
    total_high_conf_noise = n_suspected_fp + n_suspected_fn
    high_conf_noise_rate = (total_high_conf_noise / total_samples) * 100

    label_accuracy_score = ((total_samples - total_disagreements) / total_samples) * 100

    logger.info("\n" + "=" * 70)
    logger.info("📊 HASIL PENGUKURAN KUALITAS LABEL (OOF 5-FOLD CV)")
    logger.info("=" * 70)
    logger.info(f"Total Sampel Evaluasi           : {total_samples}")
    logger.info(f"Label Consistency Score (Akurasi): {label_accuracy_score:.2f}%")
    logger.info(f"Label Disagreement Rate          : {disagreement_rate:.2f}% ({total_disagreements} sampel)")
    logger.info(f"High-Confidence Noise Rate       : {high_conf_noise_rate:.2f}% ({total_high_conf_noise} sampel)")
    logger.info(f"  - Suspected False Positives (0->1): {n_suspected_fp}")
    logger.info(f"  - Suspected False Negatives (1->0): {n_suspected_fn}")

    # 4. Breakdown berdasarkan Sumber Data / Source (jika ada kolom source)
    source_summary = []
    if "source" in dev_df.columns:
        for src, group in dev_df.groupby("source"):
            src_total = len(group)
            src_disagree = group["is_disagreement"].sum()
            src_noise = (group["noise_type"] != "Clean").sum()
            source_summary.append({
                "source": src,
                "total": src_total,
                "disagreement_count": int(src_disagree),
                "disagreement_rate_pct": round((src_disagree / src_total) * 100, 2),
                "high_conf_noise_count": int(src_noise),
                "high_conf_noise_rate_pct": round((src_noise / src_total) * 100, 2)
            })

    # 5. Simpan daftar sampel berisiko salah label
    suspected_df = dev_df[dev_df["is_disagreement"] == 1].sort_values(by="ce_loss", ascending=False)
    output_csv = OUTPUT_DIR / "suspected_mislabeled_samples.csv"
    suspected_df.to_csv(output_csv, index=False)
    logger.info(f"💾 File sampel berisiko disimpan ke: {output_csv}")

    # 6. Buat Laporan Markdown
    report_md = OUTPUT_DIR / "label_quality_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# 📋 Laporan Pengukuran Kualitas Label Dataset (Out-Of-Fold 5-Fold CV)\n\n")
        f.write("Laporan ini menyajikan hasil analisis kualitas label pada dataset gabungan (**Development Set: 24.951 sampel**) menggunakan metode *Out-Of-Fold* (OOF) dari 5-Fold Cross-Validation XLM-RoBERTa.\n\n")
        f.write("---\n\n")
        f.write("## 📊 1. Ringkasan Metrik Kualitas Label\n\n")
        f.write(f"* **Total Sampel Evaluasi**: {total_samples:,} sampel\n")
        f.write(f"* **Label Quality / Consistency Index**: **{label_accuracy_score:.2f}%**\n")
        f.write(f"* **Label Disagreement Rate**: **{disagreement_rate:.2f}%** ({total_disagreements:,} sampel)\n")
        f.write(f"* **High-Confidence Label Noise Rate**: **{high_conf_noise_rate:.2f}%** ({total_high_conf_noise:,} sampel)\n")
        f.write(f"  * **Suspected False Positives (Asli: Non-Bully, Prediksi OOF: Bully $\\ge 80\\%$)**: {n_suspected_fp:,} sampel\n")
        f.write(f"  * **Suspected False Negatives (Asli: Bully, Prediksi OOF: Non-Bully $\\le 20\\%$)**: {n_suspected_fn:,} sampel\n\n")
        f.write("---\n\n")
        
        if source_summary:
            f.write("## 🌐 2. Breakdown Kualitas Label per Sumber Data (*Source*)\n\n")
            f.write("| Sumber Data (*Source*) | Total Sampel | Disagreement Count | Disagreement Rate (%) | High-Conf Noise Count | High-Conf Noise Rate (%) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for row in source_summary:
                f.write(f"| **{row['source']}** | {row['total']:,} | {row['disagreement_count']:,} | {row['disagreement_rate_pct']}% | {row['high_conf_noise_count']:,} | {row['high_conf_noise_rate_pct']}% |\n")
            f.write("\n---\n\n")

        f.write("## ⚠️ 3. Top 10 Sampel Terindikasi Salah Label (Cross-Entropy Loss Tertinggi)\n\n")
        f.write("| No | Label Asli | Pred OOF (Prob Bully) | Loss | Teks Komentar |\n")
        f.write("| :---: | :---: | :---: | :---: | :--- |\n")
        top10 = suspected_df.head(10)
        for idx, (_, row) in enumerate(top10.iterrows(), 1):
            text_snippet = str(row["text"]).replace("|", "\\|").replace("\n", " ")
            if len(text_snippet) > 100:
                text_snippet = text_snippet[:100] + "..."
            label_str = "Bully (1)" if row["label"] == 1 else "Non-Bully (0)"
            f.write(f"| {idx} | {label_str} | {row['oof_pred']} ({row['oof_prob_1']:.4f}) | {row['ce_loss']:.4f} | {text_snippet} |\n")

        f.write("\n---\n\n")
        f.write("## 💡 Kesimpulan & Rekomendasi\n\n")
        f.write(f"1. **Tingkat Kebersihan Dataset**: Kualitas pelabelan dataset gabungan tergolong **sangat tinggi ({label_accuracy_score:.2f}%)**, dengan hanya {disagreement_rate:.2f}% sampel yang menunjukkan perbedaan label antara *ground truth* dan prediksi OOF.\n")
        f.write(f"2. **Derau Label Berisiko Tinggi**: Ditemukan sebanyak **{total_high_conf_noise} sampel** yang berpotensi kuat mengalami kesalahan pelabelan manual/otomatis.\n")
        f.write("3. **Tindakan Lanjutan**: Daftar lengkap sampel berisiko telah disimpan ke `outputs/suspected_mislabeled_samples.csv` untuk keperluan pemeriksaan manual (*human-in-the-loop audit*) atau koreksi otomatis.\n")

    logger.info(f"📄 Laporan kualitas label disimpan ke: {report_md}")

if __name__ == "__main__":
    run_oof_inference()
