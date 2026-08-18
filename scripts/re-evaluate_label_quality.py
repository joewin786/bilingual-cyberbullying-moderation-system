"""
re-evaluate_label_quality.py
============================
Script untuk menghitung ulang metrik kualitas label setelah audit manual dilakukan.
Membandingkan status sebelum vs sesudah audit, menerapkan koreksi label ke train.csv & val.csv,
serta memperbarui laporan ke outputs/post_audit_label_quality_report.md
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("re_evaluate_label_quality")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

AUDITED_CSV = OUTPUT_DIR / "audited_mislabeled_samples.csv"
REPORT_MD = OUTPUT_DIR / "post_audit_label_quality_report.md"

def main():
    logger.info("=" * 70)
    logger.info("📊 PENGUKURAN ULANG KUALITAS LABEL DATASET (PASCA AUDIT MANUAL)")
    logger.info("=" * 70)

    if not AUDITED_CSV.exists():
        logger.error(f"File {AUDITED_CSV} tidak ditemukan. Jalankan audit manual terlebih dahulu!")
        return

    df_audit = pd.read_csv(AUDITED_CSV)

    audited_mask = df_audit["audited"] == True
    changed_mask = audited_mask & (df_audit["label"] != df_audit["new_label"])
    confirmed_mask = audited_mask & (df_audit["label"] == df_audit["new_label"])

    n_total_candidates = len(df_audit)
    n_audited = int(audited_mask.sum())
    n_changed = int(changed_mask.sum())
    n_confirmed = int(confirmed_mask.sum())
    n_remaining = n_total_candidates - n_audited

    # Hitung dampak pada 24.951 sampel Development Set
    dev_total = 24951
    orig_disagreements = 4041
    orig_consistency = (dev_total - orig_disagreements) / dev_total * 100

    new_disagreements = orig_disagreements - n_changed
    new_consistency = (dev_total - new_disagreements) / dev_total * 100

    logger.info(f"Total Sampel Kandidat Mislabeled: {n_total_candidates:,}")
    logger.info(f"Sampel Telah Diaudit Manual     : {n_audited:,}")
    logger.info(f"  - Label Dikoreksi (Diubah)    : {n_changed:,}")
    logger.info(f"  - Label Dikonfirmasi Asli     : {n_confirmed:,}")
    logger.info(f"  - Belum Diaudit               : {n_remaining:,}")
    logger.info("-" * 70)
    logger.info(f"Label Consistency Index Sebelum Audit : {orig_consistency:.2f}%")
    logger.info(f"Label Consistency Index Setelah Audit : {new_consistency:.2f}% (+{(new_consistency - orig_consistency):.2f}%)")

    # Terapkan koreksi ke train.csv & val.csv
    changed_df = df_audit[changed_mask]
    corrections = dict(zip(changed_df["text"], changed_df["new_label"]))

    applied_counts = {}
    for file_name in ["train.csv", "val.csv"]:
        csv_path = PROCESSED_DIR / file_name
        if csv_path.exists():
            df_ds = pd.read_csv(csv_path)
            
            # Backup otomatis
            backup_path = csv_path.with_name(f"{csv_path.stem}_backup_post_audit.csv")
            df_ds.to_csv(backup_path, index=False)
            
            count = 0
            for idx, row in df_ds.iterrows():
                txt = row["text"]
                if txt in corrections:
                    df_ds.at[idx, "label"] = corrections[txt]
                    count += 1
            
            df_ds.to_csv(csv_path, index=False)
            applied_counts[file_name] = count

    # Buat Laporan Markdown
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# 📋 Laporan Pengukuran Ulang Kualitas Label Dataset (Pasca Audit Manual)\n\n")
        f.write("Laporan ini menyajikan statistik pengukuran ulang kualitas label setelah proses **audit manual** dilakukan pada sampel kandidat derau label (*suspected mislabeled samples*).\n\n")
        f.write("---\n\n")
        f.write("## 📊 1. Statistik Hasil Audit Manual\n\n")
        f.write(f"* **Total Sampel Kandidat Ditinjau**: {n_total_candidates:,} sampel\n")
        f.write(f"* **Total Sampel Diaudit**: **{n_audited:,} sampel** (Progress: {((n_audited / n_total_candidates) * 100):.1f}%)\n")
        f.write(f"  * **Label Dikoreksi (Diubah)**: **{n_changed:,} sampel**\n")
        f.write(f"  * **Label Dikonfirmasi Asli**: **{n_confirmed:,} sampel**\n")
        f.write(f"  * **Sisa Belum Diaudit**: **{n_remaining:,} sampel**\n\n")
        f.write("---\n\n")
        f.write("## 📈 2. Perbandingan Kualitas Label Sebelum vs Sesudah Audit\n\n")
        f.write("| Metrik Kualitas Label | Sebelum Audit | Sesudah Audit | Selisih / Perbaikan |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Label Consistency Index** | **{orig_consistency:.2f}%** | **{new_consistency:.2f}%** | **+{(new_consistency - orig_consistency):.2f}%** |\n")
        f.write(f"| **Jumlah Disagreement Label** | {orig_disagreements:,} | {new_disagreements:,} | **-{n_changed:,} kesalahan** |\n")
        f.write(f"| **Koreksi Diterapkan di train.csv** | 0 | {applied_counts.get('train.csv', 0):,} | Applied |\n")
        f.write(f"| **Koreksi Diterapkan di val.csv** | 0 | {applied_counts.get('val.csv', 0):,} | Applied |\n\n")
        f.write("---\n\n")
        f.write("## ✏️ 3. Sampel Komentar yang Berhasil Dikoreksi\n\n")
        f.write("| No | Label Asli | Label Baru (Hasil Audit) | OOF Prob Bully | Teks Komentar |\n")
        f.write("| :---: | :---: | :---: | :---: | :--- |\n")
        
        for idx, (_, row) in enumerate(changed_df.iterrows(), 1):
            orig_lbl = "Bully (1)" if row["label"] == 1 else "Non-Bully (0)"
            new_lbl = "Bully (1)" if row["new_label"] == 1 else "Non-Bully (0)"
            txt_snippet = str(row["text"]).replace("|", "\\|").replace("\n", " ")
            if len(txt_snippet) > 90:
                txt_snippet = txt_snippet[:90] + "..."
            prob_str = f"{row['oof_prob_1']:.4f}"
            f.write(f"| {idx} | {orig_lbl} | **{new_lbl}** | {prob_str} | {txt_snippet} |\n")

        f.write("\n---\n\n")
        f.write("## 💡 Kesimpulan\n\n")
        f.write(f"1. Dengan mengoreksi **{n_changed} sampel** label yang salah, kebersihan dan konsistensi dataset meningkat dari **{orig_consistency:.2f}%** menjadi **{new_consistency:.2f}%**.\n")
        f.write("2. Seluruh koreksi telah disinkronkan ke file `train.csv` dan `val.csv` dengan pembuatan file backup otomatis `_backup_post_audit.csv`.\n")

    logger.info(f"📄 Laporan pengukuran ulang disimpan ke: {REPORT_MD}")

if __name__ == "__main__":
    main()
