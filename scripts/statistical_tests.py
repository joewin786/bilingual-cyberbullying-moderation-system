"""
statistical_tests.py
========================
Uji Signifikansi Statistik untuk Sistem Deteksi Cyberbullying.

Dua uji statistik non-parametrik:
  1. Wilcoxon Signed-Rank Test — Menguji perbedaan metrik performa
     (F1 Macro, Akurasi) antar model secara berpasangan pada K-Fold CV.
  2. McNemar's Test — Menguji perbedaan pola kesalahan klasifikasi
     antara dua model pada level prediksi individual (held-out test set).

Sumber data:
  - Wilcoxon: outputs/kfold/classifiers_kfold_results.json
              outputs/kfold/transformers_kfold_results.json
              outputs/kfold/xlmr_kfold_results.json
  - McNemar:  Re-inferensi pada data/processed/test.csv menggunakan
              model yang tersimpan.

Output:
  outputs/statistical_tests/
    wilcoxon_results.json
    mcnemar_results.json
    statistical_tests_summary.txt
    wilcoxon_heatmap.png
    mcnemar_heatmap.png

Cara jalankan:
    python scripts/statistical_tests.py
    python scripts/statistical_tests.py --wilcoxon-only   # Hanya Wilcoxon
    python scripts/statistical_tests.py --mcnemar-only    # Hanya McNemar

Konfigurasi:
    α = 0.05 (tingkat signifikansi)
    Koreksi Bonferroni diterapkan untuk multiple comparisons.
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder untuk menangani tipe data NumPy."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import pandas as pd
from scipy import stats

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("statistical_tests")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KFOLD_DIR = PROJECT_ROOT / "outputs" / "kfold"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_SAVE_DIR = PROJECT_ROOT / "models"

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
ALPHA = 0.05  # Tingkat signifikansi
RANDOM_SEED = 42
MAX_LENGTH = 128


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING — K-Fold Metrics
# ══════════════════════════════════════════════════════════════
def load_kfold_metrics():
    """
    Memuat metrik per-fold dari model Transformer.
    
    Returns:
        dict: {model_name: {"val": [f1_macro_fold1, ...], "test": [f1_macro_fold1, ...],
                             "val_acc": [...], "test_acc": [...]}}
    """
    all_models = {}

    # ── 1. Transformer K-Fold results ──
    tf_path = KFOLD_DIR / "transformers_kfold_results.json"
    if tf_path.exists():
        logger.info(f"📂 Memuat transformer K-Fold results: {tf_path}")
        with open(tf_path, "r", encoding="utf-8") as f:
            tf_data = json.load(f)

        for model_entry in tf_data:
            name = model_entry["name"]
            folds = model_entry.get("folds", [])

            if folds:
                all_models[name] = {
                    "val_f1_macro": [f["val"]["f1_macro"] for f in folds],
                    "test_f1_macro": [f["test"]["f1_macro"] for f in folds],
                    "val_accuracy": [f["val"]["accuracy"] for f in folds],
                    "test_accuracy": [f["test"]["accuracy"] for f in folds],
                    "source": "transformers_kfold",
                }
                logger.info(f"  ✅ {name}: {len(folds)} folds dimuat")
    else:
        logger.warning(f"  ⚠️ File tidak ditemukan: {tf_path}")

    logger.info(f"\n📊 Total model dimuat: {len(all_models)}")
    return all_models


# ══════════════════════════════════════════════════════════════
# 2. WILCOXON SIGNED-RANK TEST
# ══════════════════════════════════════════════════════════════
def run_wilcoxon_tests(all_models, metric_key="test_f1_macro"):
    """
    Menjalankan Wilcoxon Signed-Rank Test untuk semua pasangan model.

    Args:
        all_models: dict dari load_kfold_metrics()
        metric_key: kunci metrik yang diuji (test_f1_macro, test_accuracy, dll.)

    Returns:
        list[dict]: Hasil uji per pasangan.
    """
    logger.info(f"\n{'═' * 70}")
    logger.info(f"🧪 WILCOXON SIGNED-RANK TEST (Metrik: {metric_key})")
    logger.info(f"{'═' * 70}")

    model_names = sorted(all_models.keys())
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2
    alpha_corrected = ALPHA / n_comparisons if n_comparisons > 0 else ALPHA

    logger.info(f"  Jumlah model: {n_models}")
    logger.info(f"  Jumlah perbandingan: {n_comparisons}")
    logger.info(f"  α original: {ALPHA}")
    logger.info(f"  α Bonferroni-corrected: {alpha_corrected:.6f}")

    # Peringatan: K=5 → p-value minimum Wilcoxon = 0.0625
    n_folds = len(next(iter(all_models.values()))[metric_key])
    if n_folds <= 6:
        logger.warning(
            f"  ⚠️ PERINGATAN: Dengan K={n_folds}, p-value minimum Wilcoxon "
            f"signed-rank test (two-sided) = {1 / (2 ** n_folds):.4f} × 2 = "
            f"{2 / (2 ** n_folds):.4f}. "
            f"Uji ini memiliki statistical power yang rendah untuk K kecil. "
            f"McNemar's test pada level per-sampel direkomendasikan sebagai "
            f"pelengkap."
        )

    results = []

    for model_a, model_b in combinations(model_names, 2):
        scores_a = np.array(all_models[model_a][metric_key])
        scores_b = np.array(all_models[model_b][metric_key])

        diff = scores_a - scores_b
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        mean_diff = np.mean(diff)

        # Wilcoxon signed-rank test
        # Jika semua perbedaan = 0, p-value = 1.0
        if np.all(diff == 0):
            statistic, p_value = 0.0, 1.0
        else:
            try:
                statistic, p_value = stats.wilcoxon(
                    scores_a, scores_b,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            except ValueError:
                # Terjadi jika semua perbedaan nol setelah zero_method
                statistic, p_value = 0.0, 1.0

        significant_original = bool(p_value < ALPHA)
        significant_corrected = bool(p_value < alpha_corrected)

        result = {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric_key,
            "scores_a": scores_a.tolist(),
            "scores_b": scores_b.tolist(),
            "mean_a": round(float(mean_a), 6),
            "mean_b": round(float(mean_b), 6),
            "mean_difference": round(float(mean_diff), 6),
            "wilcoxon_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "alpha": ALPHA,
            "alpha_bonferroni": round(alpha_corrected, 6),
            "significant_without_correction": bool(significant_original),
            "significant_with_bonferroni": bool(significant_corrected),
            "better_model": model_a if mean_a > mean_b else model_b,
        }
        results.append(result)

        sig_mark = "✅ SIGNIFIKAN" if significant_corrected else (
            "⚠️ Signifikan (tanpa koreksi)" if significant_original else "❌ Tidak signifikan"
        )
        logger.info(
            f"\n  {model_a} vs {model_b}:\n"
            f"    Mean A: {mean_a:.4f} | Mean B: {mean_b:.4f} | "
            f"Diff: {mean_diff:+.4f}\n"
            f"    W-stat: {statistic:.4f} | p-value: {p_value:.6f}\n"
            f"    → {sig_mark}"
        )

    return results


def create_wilcoxon_heatmap(results, model_names, metric_label="F1 Macro"):
    """Buat heatmap p-values dari Wilcoxon test."""
    n = len(model_names)
    p_matrix = np.ones((n, n))
    sig_matrix = np.full((n, n), "", dtype=object)

    name_to_idx = {name: i for i, name in enumerate(model_names)}

    for r in results:
        i = name_to_idx[r["model_a"]]
        j = name_to_idx[r["model_b"]]
        p_val = r["p_value"]
        p_matrix[i][j] = p_val
        p_matrix[j][i] = p_val

        if r["significant_with_bonferroni"]:
            sig_matrix[i][j] = "**"
            sig_matrix[j][i] = "**"
        elif r["significant_without_correction"]:
            sig_matrix[i][j] = "*"
            sig_matrix[j][i] = "*"

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), max(8, n * 1.2)))

    # Custom colormap: hijau (signifikan) → merah (tidak signifikan)
    colors_list = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
    cmap = mcolors.LinearSegmentedColormap.from_list("sig", colors_list)

    # Mask diagonal
    mask = np.eye(n, dtype=bool)
    masked_data = np.ma.array(p_matrix, mask=mask)

    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=0.1, aspect="auto")

    # Anotasi
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="gray", fontweight="bold")
            else:
                p_val = p_matrix[i][j]
                sig = sig_matrix[i][j]
                color = "white" if p_val < 0.03 else "black"
                text = f"{p_val:.4f}\n{sig}" if sig else f"{p_val:.4f}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    # Shorten names for display
    short_names = []
    for name in model_names:
        if "XLM-RoBERTa" in name and "fine-tuned" in name:
            short_names.append("XLM-R\n(fine-tuned)")
        elif "XLM-RoBERTa" in name:
            short_names.append("XLM-R\nBase")
        elif "IndoBERT" in name and "IndoBenchmark" in name:
            short_names.append("IndoBERT\n(IndoBench)")
        elif "IndoBERT" in name and "IndoLEM" in name:
            short_names.append("IndoBERT\n(IndoLEM)")
        elif "mBERT" in name:
            short_names.append("mBERT")
        elif "Cosine" in name:
            short_names.append("Cosine\nSimilarity")
        else:
            short_names.append(name)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(short_names, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("p-value", fontsize=11, fontweight="bold")

    ax.set_title(
        f"Wilcoxon Signed-Rank Test — p-values ({metric_label})\n"
        f"α = {ALPHA} | Bonferroni α = {ALPHA / max(len(results), 1):.4f}\n"
        f"** = Signifikan (Bonferroni) | * = Signifikan (tanpa koreksi)",
        fontsize=12, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "wilcoxon_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  💾 Heatmap Wilcoxon disimpan: {save_path}")


# ══════════════════════════════════════════════════════════════
# 3. McNEMAR'S TEST — Re-Inference
# ══════════════════════════════════════════════════════════════
def load_test_data():
    """Memuat test set."""
    test_path = PROCESSED_DIR / "test.csv"
    logger.info(f"📂 Memuat test set: {test_path}")
    df = pd.read_csv(test_path)
    logger.info(f"  Jumlah sampel test: {len(df)}")
    return df


def get_classifier_predictions(test_df):
    """
    Mendapatkan prediksi dari model tradisional (SVM, NB, Cosine).
    Menggunakan model comparison yang tersimpan di models/comparison/.
    """
    import joblib

    predictions = {}
    comparison_dir = MODEL_SAVE_DIR / "comparison"

    texts = test_df["text"].astype(str).tolist()

    # ── TF-IDF Models ──
    tfidf_path = comparison_dir / "tfidf_vectorizer.joblib"
    if tfidf_path.exists():
        logger.info("  🔤 Memuat TF-IDF vectorizer ...")
        tfidf = joblib.load(tfidf_path)
        X_test_tfidf = tfidf.transform(texts)

        # Naive Bayes
        nb_path = comparison_dir / "naive_bayes_model.joblib"
        if nb_path.exists():
            logger.info("  🎯 Prediksi Naive Bayes ...")
            nb_model = joblib.load(nb_path)
            predictions["Naive Bayes"] = nb_model.predict(X_test_tfidf)
            logger.info(f"    ✅ Naive Bayes: {len(predictions['Naive Bayes'])} prediksi")

        # SVM Linear
        svm_path = comparison_dir / "svm_linear_model.joblib"
        if svm_path.exists():
            logger.info("  🎯 Prediksi SVM Linear ...")
            svm_model = joblib.load(svm_path)
            predictions["SVM Linear"] = svm_model.predict(X_test_tfidf)
            logger.info(f"    ✅ SVM Linear: {len(predictions['SVM Linear'])} prediksi")
    else:
        logger.warning(f"  ⚠️ TF-IDF vectorizer tidak ditemukan: {tfidf_path}")

    # ── Cosine Similarity ──
    cosine_path = comparison_dir / "cosine_centroids.joblib"
    if cosine_path.exists():
        logger.info("  🎯 Prediksi Cosine Similarity ...")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from sklearn.preprocessing import normalize

            centroids = joblib.load(cosine_path)

            # Ekstrak embeddings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            model = AutoModel.from_pretrained("xlm-roberta-base").to(device)
            model.eval()

            all_embeddings = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                all_embeddings.append(emb.cpu().numpy())

            X_test_emb = np.vstack(all_embeddings)
            X_test_norm = normalize(X_test_emb, norm="l2")

            # Prediksi berdasarkan cosine similarity ke centroid
            preds = []
            for x in X_test_norm:
                sims = {cls: np.dot(x, centroid) for cls, centroid in centroids.items()}
                preds.append(max(sims, key=sims.get))
            predictions["Cosine Similarity"] = np.array(preds)
            logger.info(f"    ✅ Cosine Similarity: {len(predictions['Cosine Similarity'])} prediksi")

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"    ⚠️ Gagal memuat Cosine Similarity: {e}")
    else:
        logger.warning(f"  ⚠️ Cosine centroids tidak ditemukan: {cosine_path}")

    return predictions


def get_transformer_predictions(test_df):
    """
    Mendapatkan prediksi dari model Transformer yang tersimpan.
    Menggunakan fold_1 best_model dari setiap transformer.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.texts = list(texts)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                truncation=True,
                max_length=self.max_length,
            )
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            }

    predictions = {}
    texts = test_df["text"].astype(str).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model Transformer dari kfold ──
    transformer_models = {
        "XLM-RoBERTa Base": "xlm-roberta",
        "IndoBERT Base (IndoBenchmark)": "indobert-p1",
        "mBERT Base (Multilingual)": "mbert-cased",
        "IndoBERT Uncased (IndoLEM)": "indobert-indolem",
    }

    kfold_dir = MODEL_SAVE_DIR / "kfold_transformers"

    for model_name, model_key in transformer_models.items():
        # Gunakan fold_1 best_model sebagai representasi
        model_dir = kfold_dir / model_key / "fold_1" / "best_model"

        if not model_dir.exists():
            # Coba cari checkpoint
            fold_dir = kfold_dir / model_key / "fold_1"
            if fold_dir.exists():
                ckpt_dirs = [d for d in (fold_dir / "checkpoints").iterdir()
                             if d.is_dir() and (d / "config.json").exists()] \
                    if (fold_dir / "checkpoints").exists() else []
                if ckpt_dirs:
                    model_dir = sorted(ckpt_dirs, key=lambda x: x.stat().st_mtime)[-1]

        if not model_dir.exists() or not (model_dir / "config.json").exists():
            logger.warning(f"  ⚠️ Model {model_name} tidak ditemukan di {model_dir}")
            continue

        logger.info(f"  🤖 Memuat {model_name} dari {model_dir} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            model.to(device)

            dataset = SimpleDataset(texts, tokenizer, MAX_LENGTH)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(PROJECT_ROOT / "models" / "temp"),
                    per_device_eval_batch_size=32,
                    fp16=(device == "cuda"),
                    report_to="none",
                ),
                data_collator=data_collator,
            )

            preds_output = trainer.predict(dataset)
            preds = np.argmax(preds_output.predictions, axis=-1)
            predictions[model_name] = preds
            logger.info(f"    ✅ {model_name}: {len(preds)} prediksi")

            del model, tokenizer, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"    ⚠️ Gagal memuat {model_name}: {e}")

    # ── XLM-RoBERTa fine-tuned (model utama) ──
    xlmr_main_dir = MODEL_SAVE_DIR / "xlmr_cyberbully" / "best_model"
    if xlmr_main_dir.exists() and (xlmr_main_dir / "config.json").exists():
        logger.info(f"  🤖 Memuat XLM-RoBERTa (fine-tuned) dari {xlmr_main_dir} ...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(xlmr_main_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(xlmr_main_dir))
            model.to(device)

            dataset = SimpleDataset(texts, tokenizer, MAX_LENGTH)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(PROJECT_ROOT / "models" / "temp"),
                    per_device_eval_batch_size=32,
                    fp16=(device == "cuda"),
                    report_to="none",
                ),
                data_collator=data_collator,
            )

            preds_output = trainer.predict(dataset)
            preds = np.argmax(preds_output.predictions, axis=-1)
            predictions["XLM-RoBERTa (fine-tuned)"] = preds
            logger.info(f"    ✅ XLM-RoBERTa (fine-tuned): {len(preds)} prediksi")

            del model, tokenizer, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"    ⚠️ Gagal memuat XLM-RoBERTa (fine-tuned): {e}")
    else:
        logger.warning(f"  ⚠️ XLM-RoBERTa fine-tuned tidak ditemukan: {xlmr_main_dir}")

    return predictions


def run_mcnemar_test(y_true, preds_a, preds_b):
    """
    Menjalankan McNemar's test untuk dua model.

    Args:
        y_true: Label aktual
        preds_a: Prediksi model A
        preds_b: Prediksi model B

    Returns:
        dict: Hasil uji McNemar
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Tabel kontigensi 2×2
    # n11: keduanya benar
    # n12: A benar, B salah
    # n21: A salah, B benar
    # n22: keduanya salah
    n11 = int(np.sum(correct_a & correct_b))
    n12 = int(np.sum(correct_a & ~correct_b))
    n21 = int(np.sum(~correct_a & correct_b))
    n22 = int(np.sum(~correct_a & ~correct_b))

    contingency_table = [[n11, n12], [n21, n22]]

    # McNemar's test dengan koreksi kontinuitas
    # χ² = (|n12 - n21| - 1)² / (n12 + n21)
    discordant_total = n12 + n21

    if discordant_total == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        # Koreksi kontinuitas (Edwards)
        chi2 = (abs(n12 - n21) - 1) ** 2 / discordant_total
        p_value = float(stats.chi2.sf(chi2, df=1))

    # Juga hitung exact McNemar (binomial) jika discordant < 25
    if 0 < discordant_total < 25:
        # Exact McNemar menggunakan binomial test
        p_value_exact = float(stats.binom_test(
            min(n12, n21), n12 + n21, 0.5
        )) if hasattr(stats, 'binom_test') else float(
            2 * stats.binom.cdf(min(n12, n21), discordant_total, 0.5)
        )
    else:
        p_value_exact = None

    return {
        "contingency_table": contingency_table,
        "n_both_correct": n11,
        "n_a_correct_b_wrong": n12,
        "n_a_wrong_b_correct": n21,
        "n_both_wrong": n22,
        "discordant_total": discordant_total,
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "p_value_exact": round(p_value_exact, 6) if p_value_exact is not None else None,
        "method": "exact_binomial" if (0 < discordant_total < 25) else "chi2_corrected",
    }


def run_all_mcnemar_tests(test_df, all_predictions):
    """
    Menjalankan McNemar's test untuk semua pasangan model.

    Args:
        test_df: DataFrame test set
        all_predictions: dict {model_name: predictions_array}

    Returns:
        list[dict]: Hasil uji per pasangan
    """
    logger.info(f"\n{'═' * 70}")
    logger.info(f"🧪 McNEMAR'S TEST")
    logger.info(f"{'═' * 70}")

    y_true = test_df["label"].values
    model_names = sorted(all_predictions.keys())
    n_models = len(model_names)
    n_comparisons = n_models * (n_models - 1) // 2
    alpha_corrected = ALPHA / n_comparisons if n_comparisons > 0 else ALPHA

    logger.info(f"  Jumlah model dengan prediksi: {n_models}")
    logger.info(f"  Jumlah sampel test: {len(y_true)}")
    logger.info(f"  Jumlah perbandingan: {n_comparisons}")
    logger.info(f"  α original: {ALPHA}")
    logger.info(f"  α Bonferroni-corrected: {alpha_corrected:.6f}")

    results = []

    for model_a, model_b in combinations(model_names, 2):
        preds_a = all_predictions[model_a]
        preds_b = all_predictions[model_b]

        acc_a = float(np.mean(preds_a == y_true))
        acc_b = float(np.mean(preds_b == y_true))

        mcnemar_result = run_mcnemar_test(y_true, preds_a, preds_b)

        p_val = mcnemar_result["p_value"]
        significant_original = bool(p_val < ALPHA)
        significant_corrected = bool(p_val < alpha_corrected)

        result = {
            "model_a": model_a,
            "model_b": model_b,
            "accuracy_a": round(float(acc_a), 6),
            "accuracy_b": round(float(acc_b), 6),
            **mcnemar_result,
            "alpha": ALPHA,
            "alpha_bonferroni": round(alpha_corrected, 6),
            "significant_without_correction": bool(significant_original),
            "significant_with_bonferroni": bool(significant_corrected),
            "better_model": model_a if acc_a > acc_b else model_b,
        }
        results.append(result)

        sig_mark = "✅ SIGNIFIKAN" if significant_corrected else (
            "⚠️ Signifikan (tanpa koreksi)" if significant_original else "❌ Tidak signifikan"
        )
        logger.info(
            f"\n  {model_a} vs {model_b}:\n"
            f"    Acc A: {acc_a:.4f} | Acc B: {acc_b:.4f}\n"
            f"    Discordant: n12={mcnemar_result['n_a_correct_b_wrong']}, "
            f"n21={mcnemar_result['n_a_wrong_b_correct']} "
            f"(total={mcnemar_result['discordant_total']})\n"
            f"    χ²: {mcnemar_result['chi2_statistic']:.4f} | "
            f"p-value: {mcnemar_result['p_value']:.6f}\n"
            f"    → {sig_mark}"
        )

    return results


def create_mcnemar_heatmap(results, model_names):
    """Buat heatmap p-values dari McNemar's test."""
    n = len(model_names)
    p_matrix = np.ones((n, n))
    sig_matrix = np.full((n, n), "", dtype=object)

    name_to_idx = {name: i for i, name in enumerate(model_names)}

    for r in results:
        if r["model_a"] in name_to_idx and r["model_b"] in name_to_idx:
            i = name_to_idx[r["model_a"]]
            j = name_to_idx[r["model_b"]]
            p_val = r["p_value"]
            p_matrix[i][j] = p_val
            p_matrix[j][i] = p_val

            if r["significant_with_bonferroni"]:
                sig_matrix[i][j] = "**"
                sig_matrix[j][i] = "**"
            elif r["significant_without_correction"]:
                sig_matrix[i][j] = "*"
                sig_matrix[j][i] = "*"

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), max(8, n * 1.2)))

    colors_list = ["#1a9641", "#a6d96a", "#ffffbf", "#fdae61", "#d7191c"]
    cmap = mcolors.LinearSegmentedColormap.from_list("sig", colors_list)

    mask = np.eye(n, dtype=bool)
    masked_data = np.ma.array(p_matrix, mask=mask)

    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=0.1, aspect="auto")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="gray", fontweight="bold")
            else:
                p_val = p_matrix[i][j]
                sig = sig_matrix[i][j]
                color = "white" if p_val < 0.03 else "black"
                text = f"{p_val:.4f}\n{sig}" if sig else f"{p_val:.4f}"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

    # Short names
    short_names = []
    for name in model_names:
        if "XLM-RoBERTa" in name and "fine-tuned" in name:
            short_names.append("XLM-R\n(fine-tuned)")
        elif "XLM-RoBERTa" in name:
            short_names.append("XLM-R\nBase")
        elif "IndoBERT" in name and "IndoBenchmark" in name:
            short_names.append("IndoBERT\n(IndoBench)")
        elif "IndoBERT" in name and "IndoLEM" in name:
            short_names.append("IndoBERT\n(IndoLEM)")
        elif "mBERT" in name:
            short_names.append("mBERT")
        elif "Cosine" in name:
            short_names.append("Cosine\nSimilarity")
        else:
            short_names.append(name)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(short_names, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("p-value", fontsize=11, fontweight="bold")

    n_comparisons = max(len(results), 1)
    ax.set_title(
        f"McNemar's Test — p-values (Held-out Test Set)\n"
        f"α = {ALPHA} | Bonferroni α = {ALPHA / n_comparisons:.4f}\n"
        f"** = Signifikan (Bonferroni) | * = Signifikan (tanpa koreksi)",
        fontsize=12, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    save_path = OUTPUT_DIR / "mcnemar_heatmap.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"  💾 Heatmap McNemar disimpan: {save_path}")


# ══════════════════════════════════════════════════════════════
# 4. SUMMARY GENERATION
# ══════════════════════════════════════════════════════════════
def generate_summary(wilcoxon_results=None, mcnemar_results=None):
    """Generate ringkasan teks hasil uji statistik."""
    lines = []
    lines.append("=" * 80)
    lines.append("HASIL UJI SIGNIFIKANSI STATISTIK — SISTEM DETEKSI CYBERBULLYING")
    lines.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Tingkat Signifikansi (α): {ALPHA}")
    lines.append("=" * 80)

    # ── Wilcoxon ──
    if wilcoxon_results:
        n_comp = len(wilcoxon_results)
        alpha_bonf = ALPHA / n_comp if n_comp > 0 else ALPHA

        lines.append(f"\n{'─' * 80}")
        lines.append(f"  1. WILCOXON SIGNED-RANK TEST (Paired K-Fold, Metrik: F1 Macro Test Set)")
        lines.append(f"     Jumlah perbandingan: {n_comp}")
        lines.append(f"     Koreksi Bonferroni: α = {alpha_bonf:.6f}")
        lines.append(f"{'─' * 80}")
        lines.append("")
        lines.append(f"  {'Model A':<30} {'Model B':<30} {'Mean A':>8} {'Mean B':>8} "
                      f"{'Diff':>8} {'W-stat':>8} {'p-value':>10} {'Keputusan'}")
        lines.append("  " + "-" * 130)

        for r in sorted(wilcoxon_results, key=lambda x: x["p_value"]):
            if r["significant_with_bonferroni"]:
                decision = "SIGNIFIKAN **"
            elif r["significant_without_correction"]:
                decision = "Signifikan *"
            else:
                decision = "Tidak Sig."

            lines.append(
                f"  {r['model_a']:<30} {r['model_b']:<30} "
                f"{r['mean_a']:>8.4f} {r['mean_b']:>8.4f} "
                f"{r['mean_difference']:>+8.4f} "
                f"{r['wilcoxon_statistic']:>8.1f} "
                f"{r['p_value']:>10.6f} "
                f"{decision}"
            )

        # Ringkasan signifikansi Wilcoxon
        n_sig_bonf = sum(1 for r in wilcoxon_results if r["significant_with_bonferroni"])
        n_sig_orig = sum(1 for r in wilcoxon_results if r["significant_without_correction"])
        lines.append("")
        lines.append(f"  Ringkasan: {n_sig_bonf}/{n_comp} pasangan signifikan "
                      f"(Bonferroni), {n_sig_orig}/{n_comp} (tanpa koreksi)")

        # Catatan K kecil
        any_scores = next((r["scores_a"] for r in wilcoxon_results), [])
        k = len(any_scores)
        if k <= 6:
            lines.append(f"")
            lines.append(f"  ⚠️ CATATAN PENTING: Dengan K={k} fold, p-value minimum")
            lines.append(f"     Wilcoxon signed-rank test (two-sided) adalah ~{2/(2**k):.4f}.")
            lines.append(f"     Artinya, uji ini TIDAK MUNGKIN mencapai signifikansi")
            lines.append(f"     pada α={ALPHA} dengan hanya {k} pasangan data.")
            lines.append(f"     Hasil Wilcoxon tetap dilaporkan untuk kelengkapan,")
            lines.append(f"     namun McNemar's test (per-sampel) lebih informatif")
            lines.append(f"     untuk menentukan signifikansi statistik.")

    # ── McNemar ──
    if mcnemar_results:
        n_comp = len(mcnemar_results)
        alpha_bonf = ALPHA / n_comp if n_comp > 0 else ALPHA

        lines.append(f"\n{'─' * 80}")
        lines.append(f"  2. McNEMAR'S TEST (Per-Sampel pada Held-out Test Set)")
        lines.append(f"     Jumlah perbandingan: {n_comp}")
        lines.append(f"     Koreksi Bonferroni: α = {alpha_bonf:.6f}")
        lines.append(f"{'─' * 80}")
        lines.append("")
        lines.append(f"  {'Model A':<30} {'Model B':<30} {'n12':>5} {'n21':>5} "
                      f"{'χ²':>8} {'p-value':>10} {'Keputusan'}")
        lines.append("  " + "-" * 115)

        for r in sorted(mcnemar_results, key=lambda x: x["p_value"]):
            if r["significant_with_bonferroni"]:
                decision = "SIGNIFIKAN **"
            elif r["significant_without_correction"]:
                decision = "Signifikan *"
            else:
                decision = "Tidak Sig."

            lines.append(
                f"  {r['model_a']:<30} {r['model_b']:<30} "
                f"{r['n_a_correct_b_wrong']:>5} {r['n_a_wrong_b_correct']:>5} "
                f"{r['chi2_statistic']:>8.2f} "
                f"{r['p_value']:>10.6f} "
                f"{decision}"
            )

        n_sig_bonf = sum(1 for r in mcnemar_results if r["significant_with_bonferroni"])
        n_sig_orig = sum(1 for r in mcnemar_results if r["significant_without_correction"])
        lines.append("")
        lines.append(f"  Ringkasan: {n_sig_bonf}/{n_comp} pasangan signifikan "
                      f"(Bonferroni), {n_sig_orig}/{n_comp} (tanpa koreksi)")

    # ── Keterangan ──
    lines.append(f"\n{'─' * 80}")
    lines.append("  KETERANGAN:")
    lines.append("  **  = Signifikan setelah koreksi Bonferroni (klaim kuat)")
    lines.append("  *   = Signifikan tanpa koreksi (klaim lebih lemah)")
    lines.append(f"  α   = {ALPHA} (tingkat signifikansi)")
    lines.append("  n12 = Model A benar, Model B salah (discordant pair)")
    lines.append("  n21 = Model A salah, Model B benar (discordant pair)")
    lines.append("  W-stat = Wilcoxon test statistic")
    lines.append("  χ²  = McNemar chi-squared statistic (dengan koreksi kontinuitas)")
    lines.append("=" * 80)

    summary_text = "\n".join(lines)

    summary_path = OUTPUT_DIR / "statistical_tests_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info(f"\n📋 Summary disimpan: {summary_path}")

    try:
        print("\n" + summary_text)
    except UnicodeEncodeError:
        safe_text = summary_text.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_text)

    return summary_text


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Uji Signifikansi Statistik — Cyberbullying Detection"
    )
    parser.add_argument("--wilcoxon-only", action="store_true",
                        help="Hanya jalankan Wilcoxon Signed-Rank Test")
    parser.add_argument("--mcnemar-only", action="store_true",
                        help="Hanya jalankan McNemar's Test")
    args = parser.parse_args()

    run_wilcoxon = not args.mcnemar_only
    run_mcnemar = not args.wilcoxon_only

    logger.info("\n" + "═" * 70)
    logger.info("🧪 UJI SIGNIFIKANSI STATISTIK — SISTEM DETEKSI CYBERBULLYING")
    logger.info(f"   Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   α = {ALPHA}")
    logger.info(f"   Wilcoxon: {'Ya' if run_wilcoxon else 'Tidak'}")
    logger.info(f"   McNemar:  {'Ya' if run_mcnemar else 'Tidak'}")
    logger.info("═" * 70)

    total_start = time.time()
    wilcoxon_results = None
    mcnemar_results = None

    # ═══════════════════════════════════════════════════════════
    # WILCOXON SIGNED-RANK TEST
    # ═══════════════════════════════════════════════════════════
    if run_wilcoxon:
        all_models = load_kfold_metrics()

        if len(all_models) < 2:
            logger.error("❌ Minimal 2 model diperlukan untuk Wilcoxon test!")
        else:
            # Uji pada F1 Macro (Test Set)
            wilcoxon_results = run_wilcoxon_tests(all_models, metric_key="test_f1_macro")

            # Simpan JSON
            wilcoxon_json = {
                "timestamp": datetime.now().isoformat(),
                "test_type": "Wilcoxon Signed-Rank Test",
                "metric": "test_f1_macro",
                "alpha": ALPHA,
                "n_comparisons": len(wilcoxon_results),
                "bonferroni_alpha": ALPHA / max(len(wilcoxon_results), 1),
                "results": wilcoxon_results,
            }
            json_path = OUTPUT_DIR / "wilcoxon_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(wilcoxon_json, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"  💾 Wilcoxon JSON disimpan: {json_path}")

            # Heatmap
            model_names = sorted(all_models.keys())
            create_wilcoxon_heatmap(wilcoxon_results, model_names, metric_label="F1 Macro (Test Set)")

    # ═══════════════════════════════════════════════════════════
    # McNEMAR'S TEST
    # ═══════════════════════════════════════════════════════════
    if run_mcnemar:
        test_df = load_test_data()

        logger.info("\n📥 Mengumpulkan prediksi per-sampel dari semua model ...")

        all_predictions = {}

        # Prediksi transformer
        logger.info("\n── Model Transformer ──")
        tf_preds = get_transformer_predictions(test_df)
        all_predictions.update(tf_preds)

        logger.info(f"\n📊 Total model dengan prediksi: {len(all_predictions)}")
        for name, preds in all_predictions.items():
            acc = np.mean(preds == test_df["label"].values)
            logger.info(f"  {name}: accuracy = {acc:.4f}")

        if len(all_predictions) < 2:
            logger.error("❌ Minimal 2 model diperlukan untuk McNemar's test!")
        else:
            mcnemar_results = run_all_mcnemar_tests(test_df, all_predictions)

            # Simpan JSON
            mcnemar_json = {
                "timestamp": datetime.now().isoformat(),
                "test_type": "McNemar's Test",
                "test_set_size": len(test_df),
                "alpha": ALPHA,
                "n_comparisons": len(mcnemar_results),
                "bonferroni_alpha": ALPHA / max(len(mcnemar_results), 1),
                "models": list(all_predictions.keys()),
                "results": mcnemar_results,
            }
            json_path = OUTPUT_DIR / "mcnemar_results.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mcnemar_json, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"  💾 McNemar JSON disimpan: {json_path}")

            # Heatmap
            model_names = sorted(all_predictions.keys())
            create_mcnemar_heatmap(mcnemar_results, model_names)

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    generate_summary(wilcoxon_results, mcnemar_results)

    total_time = time.time() - total_start
    logger.info(f"\n{'═' * 70}")
    logger.info("✅ UJI SIGNIFIKANSI STATISTIK SELESAI!")
    logger.info(f"   Total waktu: {total_time:.1f}s ({total_time / 60:.1f} menit)")
    logger.info(f"   Output: {OUTPUT_DIR}")
    logger.info("═" * 70)


if __name__ == "__main__":
    main()
