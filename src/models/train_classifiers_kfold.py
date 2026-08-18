"""
train_classifiers_kfold.py
============================
Stratified 5-Fold Cross-Validation untuk classifier tradisional:
  1. Naive Bayes (Multinomial + TF-IDF)
  2. Cosine Similarity (Centroid-based + XLM-R Embeddings)
  3. SVM Linear (TF-IDF)
  4. SVM RBF (XLM-R Embeddings)

Juga membandingkan dengan hasil XLM-RoBERTa K-Fold jika tersedia.

Alur:
  1. Gabung train.csv + val.csv → "development set"
  2. Pisahkan test.csv sebagai held-out test set
  3. Ekstrak XLM-R embeddings sekali untuk development + test
  4. StratifiedKFold(K=5):
     - TF-IDF: re-fit pada fold_train, transform fold_val & test
     - Embeddings: subset dari pre-computed
     - Train & evaluate setiap classifier per fold
  5. Agregasi: mean ± std per classifier
  6. Visualisasi: bar chart dengan error bars

Output:
  outputs/kfold/classifiers_kfold_results.json
  outputs/kfold/classifiers_kfold_summary.txt
  outputs/kfold/kfold_comparison_chart.png

Cara jalankan:
    python src/models/train_classifiers_kfold.py

Konfigurasi di configs/training_config.yaml
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("classifiers_kfold")

# ──────────────────────────────────────────────────────────────
# Paths & Config
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "training_config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "kfold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Konfigurasi
XLMR_MODEL_NAME    = "xlm-roberta-base"
MAX_LENGTH         = 128
EMBEDDING_BATCH_SIZE = 32
RANDOM_SEED        = cfg.get("data", {}).get("random_seed", 42)
TFIDF_MAX_FEATURES = 20000
LABEL_NAMES        = ["non-bully", "bully"]


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_data():
    """Load & gabung train+val sebagai development set, test sebagai held-out."""
    logger.info("=" * 60)
    logger.info("📂 Memuat dataset ...")

    data_cfg = cfg["data"]
    train_df = pd.read_csv(PROCESSED_DIR / data_cfg["train_file"])
    val_df   = pd.read_csv(PROCESSED_DIR / data_cfg["val_file"])
    test_df  = pd.read_csv(PROCESSED_DIR / data_cfg["test_file"])

    # Gabung train+val → development set
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    dev_df = dev_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    for name, df in [("Dev (train+val)", dev_df), ("Test (held-out)", test_df)]:
        dist = df["label"].value_counts().to_dict()
        logger.info(f"  {name}: {len(df)} samples — bully={dist.get(1, 0)}, non-bully={dist.get(0, 0)}")

    return dev_df, test_df


# ══════════════════════════════════════════════════════════════
# 2. XLM-R EMBEDDINGS
# ══════════════════════════════════════════════════════════════
def extract_xlmr_embeddings(texts, tokenizer, model, device, batch_size=EMBEDDING_BATCH_SIZE):
    """Ekstrak mean-pooled embeddings dari XLM-RoBERTa."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            masked_embeddings = token_embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / counts

        all_embeddings.append(mean_embeddings.cpu().numpy())

        if (i // batch_size + 1) % 20 == 0:
            logger.info(f"    Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    return np.vstack(all_embeddings)


def get_all_embeddings(dev_texts, test_texts):
    """Ekstrak embeddings sekali untuk development + test set."""
    logger.info("\n🧠 Ekstraksi embeddings XLM-RoBERTa (sekali untuk semua fold) ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)
    model = AutoModel.from_pretrained(XLMR_MODEL_NAME).to(device)

    logger.info("  Mengekstrak embeddings development set ...")
    E_dev = extract_xlmr_embeddings(dev_texts, tokenizer, model, device)
    logger.info("  Mengekstrak embeddings test set ...")
    E_test = extract_xlmr_embeddings(test_texts, tokenizer, model, device)

    logger.info(f"  Embedding shape: dev={E_dev.shape}, test={E_test.shape}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return E_dev, E_test


# ══════════════════════════════════════════════════════════════
# 3. EVALUATION HELPER
# ══════════════════════════════════════════════════════════════
def evaluate_predictions(y_true, y_pred):
    """Evaluasi dan return dict metrik."""
    acc = accuracy_score(y_true, y_pred)

    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    _, _, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
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


# ══════════════════════════════════════════════════════════════
# 4. COSINE SIMILARITY CLASSIFIER
# ══════════════════════════════════════════════════════════════
class CosineSimilarityClassifier:
    """Classifier berdasarkan Cosine Similarity ke centroid kelas."""

    def __init__(self):
        self.centroids = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = sorted(set(y))
        for cls in self.classes:
            mask = np.array(y) == cls
            class_embeddings = X[mask]
            centroid = class_embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            self.centroids[cls] = centroid

    def predict(self, X):
        X_norm = normalize(X, norm="l2")
        predictions = []
        for x in X_norm:
            similarities = {cls: np.dot(x, centroid) for cls, centroid in self.centroids.items()}
            predictions.append(max(similarities, key=similarities.get))
        return np.array(predictions)


# ══════════════════════════════════════════════════════════════
# 5. CLASSIFIER DEFINITIONS
# ══════════════════════════════════════════════════════════════
def get_classifiers():
    """Return dict of classifier configs: name → (constructor, feature_type)."""
    return {
        "Naive Bayes": {
            "constructor": lambda: MultinomialNB(alpha=1.0),
            "feature": "tfidf",
        },
        "SVM Linear": {
            "constructor": lambda: CalibratedClassifierCV(
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=RANDOM_SEED,
                ),
                cv=3,
            ),
            "feature": "tfidf",
        },
        "SVM RBF": {
            "constructor": lambda: SVC(
                C=1.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                probability=True,
            ),
            "feature": "embedding",
        },
        "Cosine Similarity": {
            "constructor": lambda: CosineSimilarityClassifier(),
            "feature": "embedding",
        },
    }


# ══════════════════════════════════════════════════════════════
# 6. AGREGASI METRIK
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# 7. VISUALIZATION — BAR CHART WITH ERROR BARS
# ══════════════════════════════════════════════════════════════
def plot_kfold_comparison(all_classifier_results, split="test"):
    """Buat bar chart perbandingan dengan error bars (std)."""
    logger.info(f"\n📊 Membuat K-Fold comparison chart ({split}) ...")

    classifiers = []
    metrics_data = {}
    metric_names = ["accuracy", "f1_binary", "f1_macro", "precision_binary", "recall_binary"]

    for name, data in all_classifier_results.items():
        agg_key = f"{split}_aggregate"
        if agg_key not in data:
            continue
        classifiers.append(name)
        agg = data[agg_key]
        for m in metric_names:
            if m not in metrics_data:
                metrics_data[m] = {"means": [], "stds": []}
            metrics_data[m]["means"].append(agg[m]["mean"])
            metrics_data[m]["stds"].append(agg[m]["std"])

    if not classifiers:
        logger.warning("  ⚠️ Tidak ada data untuk diplot.")
        return

    x = np.arange(len(classifiers))
    width = 0.15
    n_metrics = len(metric_names)

    fig, ax = plt.subplots(figsize=(max(14, len(classifiers) * 3), 7))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    labels = ["Accuracy", "F1 Binary", "F1 Macro", "Precision", "Recall"]

    for i, (m_name, label, color) in enumerate(zip(metric_names, labels, colors)):
        offset = (i - n_metrics / 2 + 0.5) * width
        means = metrics_data[m_name]["means"]
        stds = metrics_data[m_name]["stds"]

        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=label, color=color,
            edgecolor="white", linewidth=0.5,
            alpha=0.85,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )

        # Labels di atas bar
        for bar, mean_val in zip(bars, means):
            ax.annotate(
                f"{mean_val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
            )

    ax.set_xlabel("Classifier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score (mean ± std)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"K-Fold Cross-Validation — Cyberbullying Detection ({split.capitalize()} Set)\n"
        f"K={cfg.get('kfold', {}).get('n_splits', 5)} folds, mean ± std",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, fontsize=10, rotation=15, ha="right")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"kfold_comparison_chart_{split}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  💾 Chart disimpan: {chart_path}")


# ══════════════════════════════════════════════════════════════
# 8. GENERATE SUMMARY
# ══════════════════════════════════════════════════════════════
def generate_summary(all_classifier_results, n_splits):
    """Generate summary teks."""
    lines = []
    lines.append("=" * 75)
    lines.append("CLASSIFIER K-FOLD CROSS-VALIDATION SUMMARY")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"K-Folds: {n_splits}")
    lines.append(f"Seed: {RANDOM_SEED}")
    lines.append("=" * 75)

    for split_name in ["validation", "test"]:
        lines.append(f"\n{'─' * 75}")
        lines.append(f"  {split_name.upper()} SET — Aggregated Results (mean ± std)")
        lines.append(f"{'─' * 75}")

        header = (
            f"  {'Classifier':<25} {'Feature':<12} "
            f"{'Acc':>10} {'F1_mac':>10} {'F1_bin':>10} "
            f"{'Prec':>10} {'Rec':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * 87)

        # Sort by F1 macro descending
        sorted_items = sorted(
            all_classifier_results.items(),
            key=lambda x: x[1].get(f"{split_name}_aggregate", {}).get("f1_macro", {}).get("mean", 0),
            reverse=True,
        )

        for rank, (name, data) in enumerate(sorted_items, 1):
            agg_key = f"{split_name}_aggregate"
            if agg_key not in data:
                continue
            agg = data[agg_key]
            feature = data.get("feature_type", "?")
            marker = " ★" if rank == 1 else ""

            def fmt(key):
                return f"{agg[key]['mean']:.4f}±{agg[key]['std']:.3f}"

            line = (
                f"  {name:<25} {feature:<12} "
                f"{fmt('accuracy'):>10} {fmt('f1_macro'):>10} {fmt('f1_binary'):>10} "
                f"{fmt('precision_binary'):>10} {fmt('recall_binary'):>10}{marker}"
            )
            lines.append(line)

    lines.append("\n" + "=" * 75)

    summary_text = "\n".join(lines)

    summary_path = OUTPUT_DIR / "classifiers_kfold_summary.txt"
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
# 9. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    kfold_cfg = cfg.get("kfold", {})
    n_splits  = kfold_cfg.get("n_splits", 5)
    seed      = kfold_cfg.get("random_seed", RANDOM_SEED)

    logger.info("\n" + "═" * 60)
    logger.info("🚀 CLASSIFIER K-FOLD CROSS-VALIDATION PIPELINE")
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   K-Folds: {n_splits}")
    logger.info(f"   Seed: {seed}")
    logger.info("═" * 60)

    total_start = time.time()

    # ── 1. Load Data ─────────────────────────────────────────
    dev_df, test_df = load_data()

    dev_texts  = dev_df["text"].astype(str).tolist()
    dev_labels = dev_df["label"].values
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].values

    # ── 2. Ekstrak XLM-R Embeddings (sekali) ─────────────────
    E_dev, E_test = get_all_embeddings(dev_texts, test_texts)

    # ── 3. K-Fold Cross-Validation ───────────────────────────
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    classifiers_config = get_classifiers()

    # Skip SVM RBF jika dataset terlalu besar
    if len(dev_texts) > 10000:
        logger.info("\n⚠️ Skipping SVM RBF karena dev dataset > 10,000 sampel.")
        classifiers_config.pop("SVM RBF", None)

    # Init storage per classifier
    all_results = {}
    for clf_name, clf_cfg in classifiers_config.items():
        all_results[clf_name] = {
            "feature_type": "TF-IDF" if clf_cfg["feature"] == "tfidf" else "XLM-R Embed",
            "val_fold_metrics":  [],
            "test_fold_metrics": [],
        }

    # ── Iterate over folds ───────────────────────────────────
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(dev_texts, dev_labels)):
        logger.info(f"\n{'═' * 60}")
        logger.info(f"📂 FOLD {fold_idx + 1}/{n_splits}")
        logger.info(f"{'═' * 60}")

        fold_train_texts  = [dev_texts[i] for i in train_indices]
        fold_train_labels = dev_labels[train_indices]
        fold_val_texts    = [dev_texts[i] for i in val_indices]
        fold_val_labels   = dev_labels[val_indices]

        logger.info(f"  Train: {len(fold_train_texts)} | Val: {len(fold_val_texts)}")

        # ── TF-IDF: re-fit per fold ─────────────────────────
        logger.info("  🔤 Fitting TF-IDF untuk fold ini ...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X_train_tfidf = tfidf_vectorizer.fit_transform(fold_train_texts)
        X_val_tfidf   = tfidf_vectorizer.transform(fold_val_texts)
        X_test_tfidf  = tfidf_vectorizer.transform(test_texts)

        # ── Embeddings: subset dari pre-computed ─────────────
        E_train_fold = E_dev[train_indices]
        E_val_fold   = E_dev[val_indices]

        # ── Train & Evaluate setiap classifier ───────────────
        for clf_name, clf_cfg in classifiers_config.items():
            logger.info(f"\n  🎯 {clf_name} (Fold {fold_idx + 1}) ...")

            start = time.time()
            model = clf_cfg["constructor"]()

            if clf_cfg["feature"] == "tfidf":
                model.fit(X_train_tfidf, fold_train_labels)
                y_pred_val  = model.predict(X_val_tfidf)
                y_pred_test = model.predict(X_test_tfidf)
            else:  # embedding
                model.fit(E_train_fold, fold_train_labels)
                y_pred_val  = model.predict(E_val_fold)
                y_pred_test = model.predict(E_test)

            train_time = time.time() - start

            # Evaluasi validation
            val_metrics = evaluate_predictions(fold_val_labels, y_pred_val)
            all_results[clf_name]["val_fold_metrics"].append(val_metrics)

            # Evaluasi test
            test_metrics = evaluate_predictions(test_labels, y_pred_test)
            all_results[clf_name]["test_fold_metrics"].append(test_metrics)

            logger.info(
                f"    Val  → Acc: {val_metrics['accuracy']:.4f} | "
                f"F1_mac: {val_metrics['f1_macro']:.4f} | "
                f"F1_bin: {val_metrics['f1_binary']:.4f} "
                f"({train_time:.2f}s)"
            )
            logger.info(
                f"    Test → Acc: {test_metrics['accuracy']:.4f} | "
                f"F1_mac: {test_metrics['f1_macro']:.4f} | "
                f"F1_bin: {test_metrics['f1_binary']:.4f}"
            )

    # ── 4. Agregasi ──────────────────────────────────────────
    logger.info(f"\n{'═' * 60}")
    logger.info(f"📊 AGREGASI HASIL {n_splits}-FOLD")
    logger.info(f"{'═' * 60}")

    for clf_name, data in all_results.items():
        data["validation_aggregate"] = aggregate_metrics(data["val_fold_metrics"])
        data["test_aggregate"]       = aggregate_metrics(data["test_fold_metrics"])

        val_agg = data["validation_aggregate"]
        test_agg = data["test_aggregate"]

        logger.info(f"\n  {clf_name}:")
        logger.info(
            f"    Val  F1 Macro: {val_agg['f1_macro']['mean']:.4f} ± {val_agg['f1_macro']['std']:.4f}"
        )
        logger.info(
            f"    Test F1 Macro: {test_agg['f1_macro']['mean']:.4f} ± {test_agg['f1_macro']['std']:.4f}"
        )

    # ── 5. Load XLM-R K-Fold results (jika ada) ─────────────
    xlmr_kfold_path = OUTPUT_DIR / "xlmr_kfold_results.json"
    if xlmr_kfold_path.exists():
        logger.info("\n📦 Loading XLM-RoBERTa K-Fold results ...")
        with open(xlmr_kfold_path, "r", encoding="utf-8") as f:
            xlmr_data = json.load(f)

        all_results["XLM-RoBERTa (fine-tuned)"] = {
            "feature_type":         "Transformer",
            "val_fold_metrics":     xlmr_data.get("validation_per_fold", []),
            "test_fold_metrics":    xlmr_data.get("test_per_fold", []),
            "validation_aggregate": xlmr_data.get("validation_aggregate", {}),
            "test_aggregate":       xlmr_data.get("test_aggregate", {}),
        }
        logger.info("  ✅ XLM-RoBERTa K-Fold results dimuat.")
    else:
        logger.info(f"\n  ℹ️ XLM-R K-Fold results belum ada ({xlmr_kfold_path}). Jalankan train_xlmr_kfold.py terlebih dahulu.")

    # ── 6. Visualisasi ───────────────────────────────────────
    for split in ["validation", "test"]:
        plot_kfold_comparison(all_results, split=split)

    # ── 7. Generate Summary ──────────────────────────────────
    generate_summary(all_results, n_splits)

    # ── 8. Save JSON Results ─────────────────────────────────
    # Convert untuk JSON serialization
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "n_splits":  n_splits,
        "seed":      seed,
        "data": {
            "development_samples": len(dev_df),
            "test_samples":        len(test_df),
        },
        "tfidf_config": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range":  [1, 2],
        },
        "embedding_model": XLMR_MODEL_NAME,
        "classifiers": {},
    }

    for clf_name, data in all_results.items():
        json_results["classifiers"][clf_name] = {
            "feature_type":         data["feature_type"],
            "validation_per_fold":  data["val_fold_metrics"],
            "test_per_fold":        data["test_fold_metrics"],
            "validation_aggregate": data.get("validation_aggregate", {}),
            "test_aggregate":       data.get("test_aggregate", {}),
        }

    json_path = OUTPUT_DIR / "classifiers_kfold_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n💾 JSON results disimpan: {json_path}")

    # ── Summary ──────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("\n" + "═" * 60)
    logger.info("✅ CLASSIFIER K-FOLD CROSS-VALIDATION SELESAI!")
    logger.info(f"   Total waktu: {total_time:.1f}s ({total_time / 60:.1f} menit)")
    logger.info(f"   Output: {OUTPUT_DIR}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
