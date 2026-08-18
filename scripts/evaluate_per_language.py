import json
import logging
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_xlmr import CyberbullyDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_bias")

def calculate_detailed_metrics(y_true, y_pred, probs):
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    
    return {
        "accuracy": acc,
        "precision_bully": p_bin,
        "recall_bully": r_bin,
        "f1_bully": f1_bin,
        "precision_macro": p_mac,
        "recall_macro": r_mac,
        "f1_macro": f1_mac,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
    }

def run_language_evaluation():
    model_dir = PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model"
    results_dir = PROJECT_ROOT / "outputs"
    
    if not model_dir.exists():
        logger.error(f"Model tidak ditemukan di {model_dir}")
        return

    # Load threshold dari training_results.json
    results_path = results_dir / "training_results.json"
    best_threshold = 0.50
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)
        best_threshold = res_data.get("threshold_tuning", {}).get("best_threshold", 0.50)
        logger.info(f"Loaded tuned threshold: {best_threshold}")
    
    logger.info("Memuat model dan tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model dimuat di device: {device}")
    
    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "temp"),
        per_device_eval_batch_size=32,
        fp16=(device == "cuda"),
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # Paths untuk dataset evaluasi terpisah
    datasets = {
        "id_val": PROJECT_ROOT / "data" / "processed" / "backup_id_only" / "val.csv",
        "id_test": PROJECT_ROOT / "data" / "processed" / "backup_id_only" / "test.csv",
        "en_val": PROJECT_ROOT / "data" / "processed" / "en_val_subset.csv",
        "en_test": PROJECT_ROOT / "data" / "processed" / "en_test_subset.csv",
    }
    
    results = {}
    
    for key, path in datasets.items():
        if not path.exists():
            logger.warning(f"File tidak ditemukan: {path}")
            continue
            
        logger.info(f"Mengevaluasi {key} set ({path.name})...")
        df = pd.read_csv(path)
        
        # Validasi kolom text dan label
        if "text" not in df.columns or "label" not in df.columns:
            logger.error(f"Dataset {key} tidak memiliki kolom text/label!")
            continue
            
        dataset = CyberbullyDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=128,
        )
        
        # Predict
        predictions = trainer.predict(dataset)
        logits = predictions.predictions
        probs = softmax(logits, axis=-1)
        bully_probs = probs[:, 1]
        
        # Apply threshold
        preds = (bully_probs >= best_threshold).astype(int)
        
        # Calculate metrics
        y_true = df["label"].tolist()
        metrics = calculate_detailed_metrics(y_true, preds, bully_probs)
        results[key] = metrics
        logger.info(f"F1-Macro {key}: {metrics['f1_macro']:.4f}")
        
    # Buat laporan dalam Markdown
    generate_markdown_report(results, best_threshold)

def generate_markdown_report(results, threshold):
    report_path = PROJECT_ROOT / "outputs" / "language_bias_analysis.md"
    
    # Extract data for easy formatting
    # Val
    id_val = results.get("id_val", {})
    en_val = results.get("en_val", {})
    # Test
    id_test = results.get("id_test", {})
    en_test = results.get("en_test", {})
    
    markdown_content = f"""# Laporan Analisis Bias Bahasa (Indonesia vs Inggris)

Laporan ini menyajikan hasil evaluasi performa model **XLM-RoBERTa** secara terpisah untuk data berbahasa Indonesia (ID) dan bahasa Inggris (EN). Evaluasi ini bertujuan untuk mendeteksi apakah model mengalami bias performa akibat ketidakseimbangan jumlah data latihan (di mana data bahasa Inggris lebih dominan).

> [!NOTE]  
> Evaluasi menggunakan threshold klasifikasi optimal hasil tuning: **{threshold:.2f}**.

---

## 📊 Ringkasan Perbandingan Performa

Berikut adalah perbandingan metrik evaluasi antara Bahasa Indonesia dan Bahasa Inggris pada Validation Set dan Test Set:

### 1. Validation Set (Rasio Bahasa 1:1)

| Metrik | Indonesia (ID) | Inggris (EN) | Selisih (EN - ID) | Status Bias |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | {id_val.get('accuracy', 0):.4%} | {en_val.get('accuracy', 0):.4%} | {(en_val.get('accuracy', 0) - id_val.get('accuracy', 0)):+.4%} | {"Inggris Lebih Baik" if en_val.get('accuracy', 0) > id_val.get('accuracy', 0) else "Indonesia Lebih Baik"} |
| **F1-Score (Macro)** | {id_val.get('f1_macro', 0):.4%} | {en_val.get('f1_macro', 0):.4%} | {(en_val.get('f1_macro', 0) - id_val.get('f1_macro', 0)):+.4%} | {"Inggris Lebih Baik" if en_val.get('f1_macro', 0) > id_val.get('f1_macro', 0) else "Indonesia Lebih Baik"} |
| **F1-Score (Bully)** | {id_val.get('f1_bully', 0):.4%} | {en_val.get('f1_bully', 0):.4%} | {(en_val.get('f1_bully', 0) - id_val.get('f1_bully', 0)):+.4%} | {"Inggris Lebih Baik" if en_val.get('f1_bully', 0) > id_val.get('f1_bully', 0) else "Indonesia Lebih Baik"} |
| **Precision (Bully)** | {id_val.get('precision_bully', 0):.4%} | {en_val.get('precision_bully', 0):.4%} | {(en_val.get('precision_bully', 0) - id_val.get('precision_bully', 0)):+.4%} | - |
| **Recall (Bully)** | {id_val.get('recall_bully', 0):.4%} | {en_val.get('recall_bully', 0):.4%} | {(en_val.get('recall_bully', 0) - id_val.get('recall_bully', 0)):+.4%} | - |
| **False Positive Rate (FPR)** | {id_val.get('fpr', 0):.4%} | {en_val.get('fpr', 0):.4%} | {(en_val.get('fpr', 0) - id_val.get('fpr', 0)):+.4%} | - |
| **False Negative Rate (FNR)** | {id_val.get('fnr', 0):.4%} | {en_val.get('fnr', 0):.4%} | {(en_val.get('fnr', 0) - id_val.get('fnr', 0)):+.4%} | - |

### 2. Test Set (Rasio Bahasa 1:1)

| Metrik | Indonesia (ID) | Inggris (EN) | Selisih (EN - ID) | Status Bias |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | {id_test.get('accuracy', 0):.4%} | {en_test.get('accuracy', 0):.4%} | {(en_test.get('accuracy', 0) - id_test.get('accuracy', 0)):+.4%} | {"Inggris Lebih Baik" if en_test.get('accuracy', 0) > id_test.get('accuracy', 0) else "Indonesia Lebih Baik"} |
| **F1-Score (Macro)** | {id_test.get('f1_macro', 0):.4%} | {en_test.get('f1_macro', 0):.4%} | {(en_test.get('f1_macro', 0) - id_test.get('f1_macro', 0)):+.4%} | {"Inggris Lebih Baik" if en_test.get('f1_macro', 0) > id_test.get('f1_macro', 0) else "Indonesia Lebih Baik"} |
| **F1-Score (Bully)** | {id_test.get('f1_bully', 0):.4%} | {en_test.get('f1_bully', 0):.4%} | {(en_test.get('f1_bully', 0) - id_test.get('f1_bully', 0)):+.4%} | {"Inggris Lebih Baik" if en_test.get('f1_bully', 0) > id_test.get('f1_bully', 0) else "Indonesia Lebih Baik"} |
| **Precision (Bully)** | {id_test.get('precision_bully', 0):.4%} | {en_test.get('precision_bully', 0):.4%} | {(en_test.get('precision_bully', 0) - id_test.get('precision_bully', 0)):+.4%} | - |
| **Recall (Bully)** | {id_test.get('recall_bully', 0):.4%} | {en_test.get('recall_bully', 0):.4%} | {(en_test.get('recall_bully', 0) - id_test.get('recall_bully', 0)):+.4%} | - |
| **False Positive Rate (FPR)** | {id_test.get('fpr', 0):.4%} | {en_test.get('fpr', 0):.4%} | {(en_test.get('fpr', 0) - id_test.get('fpr', 0)):+.4%} | - |
| **False Negative Rate (FNR)** | {id_test.get('fnr', 0):.4%} | {en_test.get('fnr', 0):.4%} | {(en_test.get('fnr', 0) - id_test.get('fnr', 0)):+.4%} | - |

---

## 🔍 Detail Kebingungan Model (Confusion Matrix)

### Indonesia (ID)
*   **Validation Set**:
    *   True Non-Bully: {id_val.get('confusion_matrix', {}).get('tn', 0)} | False Bully (FP): {id_val.get('confusion_matrix', {}).get('fp', 0)}
    *   False Non-Bully (FN): {id_val.get('confusion_matrix', {}).get('fn', 0)} | True Bully (TP): {id_val.get('confusion_matrix', {}).get('tp', 0)}
*   **Test Set**:
    *   True Non-Bully: {id_test.get('confusion_matrix', {}).get('tn', 0)} | False Bully (FP): {id_test.get('confusion_matrix', {}).get('fp', 0)}
    *   False Non-Bully (FN): {id_test.get('confusion_matrix', {}).get('fn', 0)} | True Bully (TP): {id_test.get('confusion_matrix', {}).get('tp', 0)}

### Inggris (EN)
*   **Validation Set**:
    *   True Non-Bully: {en_val.get('confusion_matrix', {}).get('tn', 0)} | False Bully (FP): {en_val.get('confusion_matrix', {}).get('fp', 0)}
    *   False Non-Bully (FN): {en_val.get('confusion_matrix', {}).get('fn', 0)} | True Bully (TP): {en_val.get('confusion_matrix', {}).get('tp', 0)}
*   **Test Set**:
    *   True Non-Bully: {en_test.get('confusion_matrix', {}).get('tn', 0)} | False Bully (FP): {en_test.get('confusion_matrix', {}).get('fp', 0)}
    *   False Non-Bully (FN): {en_test.get('confusion_matrix', {}).get('fn', 0)} | True Bully (TP): {en_test.get('confusion_matrix', {}).get('tp', 0)}

---

## 💡 Temuan & Analisis Bias

### 1. Perbedaan Performa Global (Macro F1)
Jika selisih Macro F1 antara Inggris dan Indonesia berada di bawah **2% (0.02)**, maka model dapat dikategorikan memiliki performa multilingual yang **relatif seimbang**. Namun jika selisihnya lebih besar dari itu, terdapat indikasi bias performa terhadap bahasa yang dominan (Inggris).

*   Selisih F1 Macro (Val) : **{(en_val.get('f1_macro', 0) - id_val.get('f1_macro', 0)):+.4%}**
*   Selisih F1 Macro (Test): **{(en_test.get('f1_macro', 0) - id_test.get('f1_macro', 0)):+.4%}**

### 2. Analisis False Positive Rate (FPR) vs False Negative Rate (FNR)
*   **False Positive Rate (FPR)**: Seberapa sering model salah mendeteksi teks aman sebagai *cyberbullying* (False Alarm).
*   **False Negative Rate (FNR)**: Seberapa sering model meloloskan teks *cyberbullying* sebagai teks aman (Kebocoran Deteksi).
*   **Perbandingan**: Jika salah satu bahasa memiliki FNR yang jauh lebih tinggi, berarti model tersebut kurang sensitif/kesulitan mendeteksi cyberbullying pada bahasa tersebut.

---

## 🛠️ Langkah Mitigasi (Rekomendasi)

Jika ditemukan bias yang signifikan, berikut adalah langkah yang bisa diambil:
1. **Threshold Tuning Spesifik Bahasa**: Karena distribusi probabilitas output model mungkin berbeda untuk setiap bahasa, kita bisa menerapkan threshold yang berbeda. Misalnya, `threshold_id = 0.40` (agar lebih sensitif) dan `threshold_en = 0.50`.
2. **Ekspansi Data Indonesia**: Menambah porsi training data berbahasa Indonesia (menggunakan data YouTube scraped yang sudah dilabeli bersih).
3. **Data Augmentation**: Melakukan translasi balik (back-translation) dari EN ke ID untuk melatih kesamaan semantik.
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    logger.info(f"Laporan berhasil dibuat di {report_path}")

if __name__ == "__main__":
    run_language_evaluation()
