#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interactive_label_corrector.py
==============================
Script interaktif untuk meninjau dan memperbaiki label yang salah (label noise)
pada test.csv and val.csv menggunakan model XLM-RoBERTa yang sudah dilatih.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import CyberbullyingPredictor

def interactive_correction():
    print("=" * 60)
    print("🤖 INTERACTIVE DATASET LABEL CORRECTOR (XLM-RoBERTa)")
    print("=" * 60)
    
    # 1. Pilih dataset
    print("\nPilih dataset yang ingin Anda periksa:")
    print("1. Test Set (data/processed/test.csv)")
    print("2. Validation Set (data/processed/val.csv)")
    choice = input("Masukkan pilihan (1/2, default 1): ").strip()
    
    file_name = "test.csv"
    if choice == "2":
        file_name = "val.csv"
        
    csv_path = PROJECT_ROOT / "data" / "processed" / file_name
    if not csv_path.exists():
        print(f"❌ Error: Berkas tidak ditemukan di {csv_path}")
        return
        
    print(f"\n📂 Memuat data dari: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Total baris: {len(df)}")
    
    # 2. Muat model predictor
    model_dir = PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model"
    if not model_dir.exists():
        print(f"❌ Error: Model tidak ditemukan di {model_dir}")
        print("   Silakan jalankan training terlebih dahulu sebelum koreksi.")
        return
        
    print(f"🧠 Memuat model XLM-RoBERTa...")
    predictor = CyberbullyingPredictor(model_dir=model_dir)
    
    # 3. Jalankan prediksi batch untuk mencari FP dan FN
    print("🔍 Menjalankan prediksi pada seluruh data untuk mendeteksi kesalahan...")
    texts = df["text"].astype(str).tolist()
    labels_true = df["label"].astype(int).tolist()
    
    predictions = predictor.predict_batch(texts)
    
    # Kumpulkan index yang salah klasifikasi
    misclassified_indices = []
    for idx, (true_lbl, pred_res) in enumerate(zip(labels_true, predictions)):
        pred_lbl = pred_res.label_id
        if true_lbl != pred_lbl:
            misclassified_indices.append(idx)
            
    total_errors = len(misclassified_indices)
    print(f"🚨 Ditemukan {total_errors} kesalahan klasifikasi (FP + FN) dari {len(df)} total data.")
    if total_errors == 0:
        print("🎉 Selamat! Model memprediksi semua data dengan benar. Tidak ada koreksi diperlukan.")
        return
        
    print("\n💡 Petunjuk Koreksi:")
    print("   - y  : Jika label asli SALAH (Program akan membalik label asli)")
    print("   - n  : Jika label asli BENAR (Prediksi model yang salah, lewati)")
    print("   - q  : Simpan koreksi saat ini dan keluar")
    print("   - Tekan [Enter] untuk melewati (n)")
    print("-" * 60)
    
    corrected_count = 0
    skipped_count = 0
    
    try:
        for i, idx in enumerate(misclassified_indices):
            text = texts[idx]
            true_lbl = labels_true[idx]
            pred_res = predictions[idx]
            pred_lbl = pred_res.label_id
            conf = pred_res.confidence * 100
            
            true_tag = "BULLY" if true_lbl == 1 else "NON-BULLY"
            pred_tag = "BULLY" if pred_lbl == 1 else "NON-BULLY"
            error_type = "False Positive (FP)" if true_lbl == 0 else "False Negative (FN)"
            
            print(f"\n[{i+1}/{total_errors}] Tipe Kesalahan: {error_type}")
            print(f"Teks          : {text}")
            print(f"Label Asli    : {true_lbl} ({true_tag})")
            print(f"Prediksi Model: {pred_lbl} ({pred_tag}) [Confidence: {conf:.2f}%]")
            
            ans = input("Apakah label asli SALAH? (y/n/q): ").strip().lower()
            
            if ans == 'q':
                print("\n💾 Menyimpan koreksi dan keluar...")
                break
            elif ans == 'y':
                # Balik label asli
                new_lbl = 1 if true_lbl == 0 else 0
                df.at[idx, 'label'] = new_lbl
                corrected_count += 1
                print(f"✅ Label diubah menjadi: {new_lbl} ({'BULLY' if new_lbl == 1 else 'NON-BULLY'})")
            else:
                skipped_count += 1
                print("⏭️ Dilewati (Label asli dipertahankan)")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Interupsi keyboard terdeteksi. Menyimpan perubahan saat ini...")
        
    # 4. Simpan hasil
    if corrected_count > 0:
        # Buat backup sebelum menimpa
        backup_path = csv_path.with_name(f"{csv_path.stem}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_old = pd.read_csv(csv_path)
        df_old.to_csv(backup_path, index=False)
        print(f"💾 Backup file asli disimpan ke: {backup_path}")
        
        # Simpan file yang diperbarui
        df.to_csv(csv_path, index=False)
        print(f"🎉 Berhasil menyimpan {corrected_count} koreksi label ke {csv_path}!")
    else:
        print("ℹ️ Tidak ada label yang diubah. File tidak dimodifikasi.")
        
    print("\nSelesai!")
    print(f"Koreksi: {corrected_count} | Dilewati: {skipped_count}")

if __name__ == "__main__":
    interactive_correction()
