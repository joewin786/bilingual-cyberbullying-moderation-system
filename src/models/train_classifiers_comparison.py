"""
train_classifiers_comparison.py
================================
Script untuk melatih dan mengevaluasi classifier tradisional
(Naive Bayes, Cosine Similarity, SVM) dan membandingkan hasilnya
dengan model XLM-RoBERTa fine-tuned yang sudah ada.

Classifier yang dibandingkan:
  1. Naive Bayes (Multinomial)  — fitur TF-IDF
  2. Cosine Similarity (Centroid-based) — fitur XLM-R embeddings
  3. SVM Linear                — fitur TF-IDF
  4. SVM RBF                   — fitur XLM-R embeddings
  5. XLM-RoBERTa (fine-tuned)  — hasil dari training_results.json

Cara jalankan:
    python src/models/train_classifiers_comparison.py

Output:
    outputs/classifier_comparison/
        comparison_report.txt
        comparison_results.json
        comparison_chart.png
        confusion_matrices.png
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import normalize
from sklearn.calibration import CalibratedClassifierCV

import torch
from transformers import AutoTokenizer, AutoModel

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("classifier_comparison")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classifier_comparison"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "comparison"
XLMR_RESULTS_PATH = PROJECT_ROOT / "outputs" / "training_results.json"

# Buat direktori output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Konfigurasi
# ──────────────────────────────────────────────────────────────
XLMR_MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
EMBEDDING_BATCH_SIZE = 32
RANDOM_SEED = 42
TFIDF_MAX_FEATURES = 20000

# Label mapping
LABEL_NAMES = ["non-bully", "bully"]


# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_data():
    """Load train/val/test splits dari data/processed/."""
    logger.info("=" * 60)
    logger.info("📂 Memuat dataset ...")

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df["label"].value_counts().to_dict()
        logger.info(f"  {name}: {len(df)} samples — bully={dist.get(1, 0)}, non-bully={dist.get(0, 0)}")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION — TF-IDF
# ══════════════════════════════════════════════════════════════
def extract_tfidf_features(train_texts, val_texts, test_texts):
    """Ekstrak fitur TF-IDF dari teks."""
    logger.info("\n🔤 Ekstraksi fitur TF-IDF ...")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),       # unigram + bigram
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,        # log-normalization
        strip_accents="unicode",
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    logger.info(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    logger.info(f"  Feature matrix shape: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Simpan vectorizer
    joblib.dump(vectorizer, MODEL_SAVE_DIR / "tfidf_vectorizer.joblib")
    logger.info(f"  💾 TF-IDF vectorizer disimpan.")

    return X_train, X_val, X_test, vectorizer


# ══════════════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION — XLM-R EMBEDDINGS
# ══════════════════════════════════════════════════════════════
def extract_xlmr_embeddings(texts, tokenizer, model, device, batch_size=EMBEDDING_BATCH_SIZE):
    """Ekstrak mean-pooled embeddings dari XLM-RoBERTa pre-trained."""
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
            # Mean pooling over token embeddings (exclude padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            masked_embeddings = token_embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / counts

        all_embeddings.append(mean_embeddings.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"    Batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")

    return np.vstack(all_embeddings)


def get_xlmr_embeddings(train_texts, val_texts, test_texts):
    """Load XLM-R model dan ekstrak embeddings untuk semua split."""
    logger.info("\n🧠 Ekstraksi embeddings XLM-RoBERTa ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)
    model = AutoModel.from_pretrained(XLMR_MODEL_NAME).to(device)

    logger.info("  Mengekstrak embeddings train ...")
    E_train = extract_xlmr_embeddings(train_texts, tokenizer, model, device)
    logger.info("  Mengekstrak embeddings val ...")
    E_val = extract_xlmr_embeddings(val_texts, tokenizer, model, device)
    logger.info("  Mengekstrak embeddings test ...")
    E_test = extract_xlmr_embeddings(test_texts, tokenizer, model, device)

    logger.info(f"  Embedding shape: train={E_train.shape}, val={E_val.shape}, test={E_test.shape}")

    # Bersihkan GPU memory
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return E_train, E_val, E_test


# ══════════════════════════════════════════════════════════════
# 4. EVALUATION HELPER
# ══════════════════════════════════════════════════════════════
def evaluate_classifier(y_true, y_pred, classifier_name):
    """Evaluasi classifier dan return dict metrik."""
    acc = accuracy_score(y_true, y_pred)

    # Binary (pos_label=1 → bully)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )
    # Macro
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    # Weighted
    _, _, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)

    metrics = {
        "classifier": classifier_name,
        "accuracy": round(acc, 4),
        "precision_binary": round(p_bin, 4),
        "recall_binary": round(r_bin, 4),
        "f1_binary": round(f1_bin, 4),
        "precision_macro": round(p_mac, 4),
        "recall_macro": round(r_mac, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
        "confusion_matrix": cm.tolist(),
    }

    logger.info(f"\n📊 {classifier_name}")
    logger.info(f"   Accuracy:        {acc:.4f}")
    logger.info(f"   F1 Binary:       {f1_bin:.4f}")
    logger.info(f"   F1 Macro:        {f1_mac:.4f}")
    logger.info(f"   F1 Weighted:     {f1_wt:.4f}")
    logger.info(f"   Precision (bin): {p_bin:.4f}")
    logger.info(f"   Recall (bin):    {r_bin:.4f}")
    logger.info(f"\n{report}")

    return metrics


# ══════════════════════════════════════════════════════════════
# 5. CLASSIFIER: NAIVE BAYES
# ══════════════════════════════════════════════════════════════
def train_naive_bayes(X_train, y_train, X_val, y_val, X_test, y_test):
    """Latih dan evaluasi Multinomial Naive Bayes."""
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Training: Naive Bayes (Multinomial + TF-IDF)")
    logger.info("=" * 60)

    start = time.time()
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info(f"  Training time: {train_time:.2f}s")

    # Evaluasi pada validation
    y_pred_val = model.predict(X_val)
    val_metrics = evaluate_classifier(y_val, y_pred_val, "Naive Bayes (Val)")

    # Evaluasi pada test
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_classifier(y_test, y_pred_test, "Naive Bayes (Test)")

    # Simpan model
    joblib.dump(model, MODEL_SAVE_DIR / "naive_bayes_model.joblib")
    logger.info(f"  💾 Model disimpan.")

    return {
        "name": "Naive Bayes",
        "feature": "TF-IDF",
        "train_time_seconds": round(train_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 6. CLASSIFIER: COSINE SIMILARITY (Centroid-based)
# ══════════════════════════════════════════════════════════════
class CosineSimilarityClassifier:
    """
    Classifier berdasarkan Cosine Similarity ke centroid kelas.

    Cara kerja:
    1. Hitung centroid (rata-rata embedding) untuk setiap kelas dari training set
    2. Untuk prediksi, hitung cosine similarity input ke setiap centroid
    3. Prediksi = kelas dengan cosine similarity tertinggi
    """

    def __init__(self):
        self.centroids = {}
        self.classes = []

    def fit(self, X, y):
        """Hitung centroid untuk setiap kelas."""
        self.classes = sorted(set(y))
        for cls in self.classes:
            mask = np.array(y) == cls
            class_embeddings = X[mask]
            centroid = class_embeddings.mean(axis=0)
            # L2-normalize centroid untuk cosine similarity
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            self.centroids[cls] = centroid

    def predict(self, X):
        """Prediksi berdasarkan cosine similarity ke centroid."""
        # L2-normalize input
        X_norm = normalize(X, norm="l2")
        predictions = []

        for x in X_norm:
            similarities = {}
            for cls, centroid in self.centroids.items():
                sim = np.dot(x, centroid)
                similarities[cls] = sim
            pred = max(similarities, key=similarities.get)
            predictions.append(pred)

        return np.array(predictions)

    def predict_similarities(self, X):
        """Return similarity scores untuk setiap kelas."""
        X_norm = normalize(X, norm="l2")
        all_sims = []
        for x in X_norm:
            sims = [np.dot(x, self.centroids[cls]) for cls in self.classes]
            all_sims.append(sims)
        return np.array(all_sims)


def train_cosine_similarity(E_train, y_train, E_val, y_val, E_test, y_test):
    """Latih dan evaluasi Cosine Similarity classifier."""
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Training: Cosine Similarity (Centroid-based + XLM-R Embeddings)")
    logger.info("=" * 60)

    start = time.time()
    model = CosineSimilarityClassifier()
    model.fit(E_train, y_train)
    train_time = time.time() - start
    logger.info(f"  Training time: {train_time:.2f}s")

    # Evaluasi pada validation
    y_pred_val = model.predict(E_val)
    val_metrics = evaluate_classifier(y_val, y_pred_val, "Cosine Similarity (Val)")

    # Evaluasi pada test
    y_pred_test = model.predict(E_test)
    test_metrics = evaluate_classifier(y_test, y_pred_test, "Cosine Similarity (Test)")

    # Simpan centroids
    joblib.dump(model, MODEL_SAVE_DIR / "cosine_centroids.joblib")
    logger.info(f"  💾 Centroids disimpan.")

    return {
        "name": "Cosine Similarity",
        "feature": "XLM-R Embeddings",
        "train_time_seconds": round(train_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 7. CLASSIFIER: SVM LINEAR
# ══════════════════════════════════════════════════════════════
def train_svm_linear(X_train, y_train, X_val, y_val, X_test, y_test):
    """Latih dan evaluasi SVM Linear."""
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Training: SVM Linear (TF-IDF)")
    logger.info("=" * 60)

    start = time.time()
    # LinearSVC lebih cepat dari SVC(kernel='linear') untuk data besar
    # CalibratedClassifierCV untuk mendapatkan probability estimates
    base_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_SEED,
    )
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)
    train_time = time.time() - start
    logger.info(f"  Training time: {train_time:.2f}s")

    # Evaluasi pada validation
    y_pred_val = model.predict(X_val)
    val_metrics = evaluate_classifier(y_val, y_pred_val, "SVM Linear (Val)")

    # Evaluasi pada test
    y_pred_test = model.predict(X_test)
    test_metrics = evaluate_classifier(y_test, y_pred_test, "SVM Linear (Test)")

    # Simpan model
    joblib.dump(model, MODEL_SAVE_DIR / "svm_linear_model.joblib")
    logger.info(f"  💾 Model disimpan.")

    return {
        "name": "SVM Linear",
        "feature": "TF-IDF",
        "train_time_seconds": round(train_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 8. CLASSIFIER: SVM RBF
# ══════════════════════════════════════════════════════════════
def train_svm_rbf(E_train, y_train, E_val, y_val, E_test, y_test):
    """Latih dan evaluasi SVM RBF pada XLM-R embeddings."""
    logger.info("\n" + "=" * 60)
    logger.info("🎯 Training: SVM RBF (XLM-R Embeddings)")
    logger.info("=" * 60)

    start = time.time()
    model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        probability=True,    # Untuk predict_proba
    )
    model.fit(E_train, y_train)
    train_time = time.time() - start
    logger.info(f"  Training time: {train_time:.2f}s")

    # Evaluasi pada validation
    y_pred_val = model.predict(E_val)
    val_metrics = evaluate_classifier(y_val, y_pred_val, "SVM RBF (Val)")

    # Evaluasi pada test
    y_pred_test = model.predict(E_test)
    test_metrics = evaluate_classifier(y_test, y_pred_test, "SVM RBF (Test)")

    # Simpan model
    joblib.dump(model, MODEL_SAVE_DIR / "svm_rbf_model.joblib")
    logger.info(f"  💾 Model disimpan.")

    return {
        "name": "SVM RBF",
        "feature": "XLM-R Embeddings",
        "train_time_seconds": round(train_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }


# ══════════════════════════════════════════════════════════════
# 9. LOAD XLM-R FINE-TUNED RESULTS
# ══════════════════════════════════════════════════════════════
def load_xlmr_results():
    """Load hasil evaluasi XLM-RoBERTa dari training_results.json."""
    logger.info("\n" + "=" * 60)
    logger.info("📦 Loading XLM-RoBERTa fine-tuned results ...")
    logger.info("=" * 60)

    if not XLMR_RESULTS_PATH.exists():
        logger.warning(f"  ⚠️ File tidak ditemukan: {XLMR_RESULTS_PATH}")
        return None

    with open(XLMR_RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    val_m = results.get("validation_metrics", {})
    test_m = results.get("test_metrics", {})

    xlmr_data = {
        "name": "XLM-RoBERTa (fine-tuned)",
        "feature": "Transformer (fine-tuned)",
        "train_time_seconds": "N/A",
        "validation": {
            "classifier": "XLM-RoBERTa (Val)",
            "accuracy": val_m.get("test_accuracy", 0),
            "precision_binary": val_m.get("test_precision_binary", 0),
            "recall_binary": val_m.get("test_recall_binary", 0),
            "f1_binary": val_m.get("test_f1_binary", 0),
            "precision_macro": val_m.get("test_precision_macro", 0),
            "recall_macro": val_m.get("test_recall_macro", 0),
            "f1_macro": val_m.get("test_f1_macro", 0),
            "f1_weighted": val_m.get("test_f1_weighted", 0),
            "confusion_matrix": None,
        },
        "test": {
            "classifier": "XLM-RoBERTa (Test)",
            "accuracy": test_m.get("test_accuracy", 0),
            "precision_binary": test_m.get("test_precision_binary", 0),
            "recall_binary": test_m.get("test_recall_binary", 0),
            "f1_binary": test_m.get("test_f1_binary", 0),
            "precision_macro": test_m.get("test_precision_macro", 0),
            "recall_macro": test_m.get("test_recall_macro", 0),
            "f1_macro": test_m.get("test_f1_macro", 0),
            "f1_weighted": test_m.get("test_f1_weighted", 0),
            "confusion_matrix": None,
        },
    }

    # Load confusion matrix dari evaluation report jika ada
    val_report_path = PROJECT_ROOT / "outputs" / "evaluation_report_validation.txt"
    test_report_path = PROJECT_ROOT / "outputs" / "evaluation_report_test.txt"

    for path, key in [(val_report_path, "validation"), (test_report_path, "test")]:
        if path.exists():
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            for i, line in enumerate(lines):
                if "True:0" in line:
                    parts0 = line.split()
                    parts1 = lines[i + 1].split()
                    try:
                        cm = [
                            [int(parts0[1]), int(parts0[2])],
                            [int(parts1[1]), int(parts1[2])],
                        ]
                        xlmr_data[key]["confusion_matrix"] = cm
                    except (ValueError, IndexError):
                        pass

    logger.info(f"  Val Accuracy:  {xlmr_data['validation']['accuracy']}")
    logger.info(f"  Val F1 Macro:  {xlmr_data['validation']['f1_macro']}")
    logger.info(f"  Test Accuracy: {xlmr_data['test']['accuracy']}")
    logger.info(f"  Test F1 Macro: {xlmr_data['test']['f1_macro']}")

    return xlmr_data


# ══════════════════════════════════════════════════════════════
# 10. VISUALIZATION — COMPARISON CHART
# ══════════════════════════════════════════════════════════════
def plot_comparison_chart(all_results, split="test"):
    """Buat bar chart perbandingan metrik semua classifier."""
    logger.info(f"\n📊 Membuat comparison chart ({split}) ...")

    classifiers = []
    accuracy_vals = []
    f1_binary_vals = []
    f1_macro_vals = []
    precision_vals = []
    recall_vals = []

    for r in all_results:
        data = r[split]
        classifiers.append(r["name"])
        accuracy_vals.append(data["accuracy"])
        f1_binary_vals.append(data["f1_binary"])
        f1_macro_vals.append(data["f1_macro"])
        precision_vals.append(data["precision_binary"])
        recall_vals.append(data["recall_binary"])

    x = np.arange(len(classifiers))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))

    # Warna yang menarik
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    bars1 = ax.bar(x - 2 * width, accuracy_vals, width, label="Accuracy", color=colors[0], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x - width, f1_binary_vals, width, label="F1 Binary", color=colors[1], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x, f1_macro_vals, width, label="F1 Macro", color=colors[2], edgecolor="white", linewidth=0.5)
    bars4 = ax.bar(x + width, precision_vals, width, label="Precision", color=colors[3], edgecolor="white", linewidth=0.5)
    bars5 = ax.bar(x + 2 * width, recall_vals, width, label="Recall", color=colors[4], edgecolor="white", linewidth=0.5)

    # Labels di atas bar
    for bars in [bars1, bars2, bars3, bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold",
            )

    ax.set_xlabel("Classifier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Perbandingan Classifier — Cyberbullying Detection ({split.capitalize()} Set)",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"comparison_chart_{split}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  💾 Chart disimpan: {chart_path}")


# ══════════════════════════════════════════════════════════════
# 11. VISUALIZATION — CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════
def plot_confusion_matrices(all_results, split="test"):
    """Buat grid confusion matrix untuk semua classifier."""
    logger.info(f"\n📊 Membuat confusion matrices ({split}) ...")

    # Filter classifier yang punya confusion matrix
    valid_results = [r for r in all_results if r[split].get("confusion_matrix") is not None]
    n = len(valid_results)

    if n == 0:
        logger.warning("  ⚠️ Tidak ada confusion matrix untuk diplot.")
        return

    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.cm.Blues

    for idx, r in enumerate(valid_results):
        ax = axes[idx]
        cm = np.array(r[split]["confusion_matrix"])

        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight="bold")

        ax.set_title(r["name"], fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES, fontsize=8)
        ax.set_yticklabels(LABEL_NAMES, fontsize=8)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Confusion Matrices — {split.capitalize()} Set",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    cm_path = OUTPUT_DIR / f"confusion_matrices_{split}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  💾 Confusion matrices disimpan: {cm_path}")


# ══════════════════════════════════════════════════════════════
# 12. GENERATE COMPARISON REPORT
# ══════════════════════════════════════════════════════════════
def generate_comparison_report(all_results):
    """Generate laporan perbandingan teks."""
    logger.info("\n📝 Generating comparison report ...")

    lines = []
    lines.append("=" * 80)
    lines.append("LAPORAN PERBANDINGAN CLASSIFIER — CYBERBULLYING DETECTION")
    lines.append(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    for split in ["validation", "test"]:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"  {split.upper()} SET RESULTS")
        lines.append(f"{'─' * 80}")

        # Header tabel
        header = f"{'Classifier':<30} {'Feature':<20} {'Acc':>7} {'F1_bin':>7} {'F1_mac':>7} {'Prec':>7} {'Rec':>7}"
        lines.append(header)
        lines.append("-" * len(header))

        # Sort by F1 macro descending
        sorted_results = sorted(all_results, key=lambda r: r[split]["f1_macro"], reverse=True)

        for rank, r in enumerate(sorted_results, 1):
            d = r[split]
            marker = " ★" if rank == 1 else ""
            line = (
                f"{r['name']:<30} {r['feature']:<20} "
                f"{d['accuracy']:>7.4f} {d['f1_binary']:>7.4f} "
                f"{d['f1_macro']:>7.4f} {d['precision_binary']:>7.4f} "
                f"{d['recall_binary']:>7.4f}{marker}"
            )
            lines.append(line)

        # Best classifier
        best = sorted_results[0]
        lines.append(f"\n  ★ Best ({split}): {best['name']} — F1 Macro: {best[split]['f1_macro']:.4f}")

    # Confusion matrices
    lines.append(f"\n{'═' * 80}")
    lines.append("CONFUSION MATRICES (TEST SET)")
    lines.append(f"{'═' * 80}")

    for r in all_results:
        cm = r["test"].get("confusion_matrix")
        if cm is not None:
            lines.append(f"\n  {r['name']}:")
            lines.append(f"              Pred:0   Pred:1")
            lines.append(f"  True:0     {cm[0][0]:>6}   {cm[0][1]:>6}")
            lines.append(f"  True:1     {cm[1][0]:>6}   {cm[1][1]:>6}")

    lines.append(f"\n{'═' * 80}")
    lines.append("TRAINING TIME")
    lines.append(f"{'═' * 80}")

    for r in all_results:
        t = r.get("train_time_seconds", "N/A")
        lines.append(f"  {r['name']:<35} {t}s")

    lines.append("\n" + "=" * 80)

    report_text = "\n".join(lines)

    # Simpan report
    report_path = OUTPUT_DIR / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"  💾 Report disimpan: {report_path}")

    # Print to console (handle Windows encoding)
    try:
        print("\n" + report_text)
    except UnicodeEncodeError:
        # Fallback: replace non-ASCII chars for Windows console
        safe_text = report_text.encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_text)

    return report_text


# ══════════════════════════════════════════════════════════════
# 13. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    logger.info("\n" + "═" * 60)
    logger.info("🚀 CLASSIFIER COMPARISON PIPELINE")
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("═" * 60)

    total_start = time.time()

    # ── 1. Load Data ─────────────────────────────────────────
    train_df, val_df, test_df = load_data()

    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # ── 2. Feature Extraction — TF-IDF ───────────────────────
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(
        train_texts, val_texts, test_texts
    )

    # ── 3. Feature Extraction — XLM-R Embeddings ────────────
    E_train, E_val, E_test = get_xlmr_embeddings(
        train_texts, val_texts, test_texts
    )

    # ── 4. Train & Evaluate Classifiers ──────────────────────
    all_results = []

    # 4a. Naive Bayes
    nb_results = train_naive_bayes(
        X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test
    )
    all_results.append(nb_results)

    # 4b. Cosine Similarity
    cos_results = train_cosine_similarity(
        E_train, y_train, E_val, y_val, E_test, y_test
    )
    all_results.append(cos_results)

    # 4c. SVM Linear
    svm_linear_results = train_svm_linear(
        X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test
    )
    all_results.append(svm_linear_results)

    # 4d. SVM RBF (Skip if dataset is too large to prevent hanging)
    if len(train_texts) <= 10000:
        svm_rbf_results = train_svm_rbf(
            E_train, y_train, E_val, y_val, E_test, y_test
        )
        all_results.append(svm_rbf_results)
    else:
        logger.info("\n⚠️ Skipping SVM RBF because train dataset size > 10,000 to prevent long training time.")

    # ── 5. Load XLM-R Results ────────────────────────────────
    xlmr_results = load_xlmr_results()
    if xlmr_results:
        all_results.append(xlmr_results)

    # ── 6. Generate Report ───────────────────────────────────
    generate_comparison_report(all_results)

    # ── 7. Visualizations ────────────────────────────────────
    for split in ["validation", "test"]:
        plot_comparison_chart(all_results, split=split)
        plot_confusion_matrices(all_results, split=split)

    # ── 8. Save JSON Results ─────────────────────────────────
    json_results = {
        "timestamp": datetime.now().isoformat(),
        "seed": RANDOM_SEED,
        "tfidf_config": {
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": [1, 2],
        },
        "embedding_model": XLMR_MODEL_NAME,
        "data": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
        "classifiers": all_results,
    }

    json_path = OUTPUT_DIR / "comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n💾 JSON results disimpan: {json_path}")

    # ── Summary ──────────────────────────────────────────────
    total_time = time.time() - total_start
    logger.info("\n" + "═" * 60)
    logger.info("✅ CLASSIFIER COMPARISON SELESAI!")
    logger.info(f"   Total waktu: {total_time:.1f}s ({total_time / 60:.1f} menit)")
    logger.info(f"   Output: {OUTPUT_DIR}")
    logger.info(f"   Models: {MODEL_SAVE_DIR}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
