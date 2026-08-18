# =============================================================================
# DATA UNDERSTANDING - Cyberbullying Text Classification
# =============================================================================
# Mendukung dataset format: CSV dan Parquet
# Kolom utama: 'text' dan 'label'
# =============================================================================

# == Import Libraries ==========================================================
import os
import sys

# Fix encoding untuk Windows terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("cp1252", "ascii"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ── Konfigurasi Tampilan ──────────────────────────────────────────────────────
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("Set2")
FIG_TITLE_STYLE = dict(fontsize=15, fontweight="bold", pad=14)
DIVIDER = "=" * 65


def section(title: str) -> None:
    """Cetak header section yang rapi."""
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# =============================================================================
# 1. LOAD DATASET
# =============================================================================
section("1. LOAD DATASET")

# ── Sesuaikan path di bawah ini ───────────────────────────────────────────────
DATASET_PATH = "./dataset/combined_dataset.csv"          # Ganti ke path file Anda
TEXT_COL     = "clean_text"           # Nama kolom teks
LABEL_COL    = "Label"                # Nama kolom label (case-sensitive!)
# ─────────────────────────────────────────────────────────────────────────────

ext = os.path.splitext(DATASET_PATH)[-1].lower()
if ext == ".csv":
    df = pd.read_csv(DATASET_PATH)
    print(f"[OK] Berhasil load CSV: '{DATASET_PATH}'")
elif ext in (".parquet", ".pq"):
    df = pd.read_parquet(DATASET_PATH)
    print(f"[OK] Berhasil load Parquet: '{DATASET_PATH}'")
else:
    raise ValueError(f"Format file tidak didukung: '{ext}'. Gunakan .csv atau .parquet")

# Validasi kolom
assert TEXT_COL  in df.columns, f"Kolom '{TEXT_COL}' tidak ditemukan!"
assert LABEL_COL in df.columns, f"Kolom '{LABEL_COL}' tidak ditemukan!"
print(f"   Shape dataset : {df.shape[0]:,} baris × {df.shape[1]} kolom")


# =============================================================================
# 2. TAMPILAN AWAL DATASET
# =============================================================================
section("2A. 5 DATA PERTAMA")
print(df.head())

section("2B. INFO DATASET")
df.info()

section("2C. DESKRIPSI STATISTIK")
print(df.describe(include="all"))


# =============================================================================
# 3. CEK KUALITAS DATA
# =============================================================================
section("3A. MISSING VALUES PER KOLOM")
mv = df.isnull().sum().rename("Missing Count")
mv_pct = (df.isnull().mean() * 100).rename("Missing (%)")
mv_tbl = pd.concat([mv, mv_pct], axis=1)
print(mv_tbl[mv_tbl["Missing Count"] > 0].to_string() or "[OK] Tidak ada missing values.")

section("3B. JUMLAH DATA DUPLIKAT")
n_dup = df.duplicated().sum()
print(f"   Jumlah baris duplikat : {n_dup:,}")
if n_dup > 0:
    print(f"   ({n_dup / len(df) * 100:.2f}% dari total data)")

section("3C. TEKS KOSONG / HANYA WHITESPACE")
empty_mask = df[TEXT_COL].isna() | df[TEXT_COL].astype(str).str.strip().eq("")
n_empty = empty_mask.sum()
print(f"   Jumlah teks kosong / whitespace saja : {n_empty:,}")
if n_empty > 0:
    print(df[empty_mask][[TEXT_COL, LABEL_COL]])


# =============================================================================
# 4. ANALISIS DISTRIBUSI LABEL
# =============================================================================
section("4. DISTRIBUSI LABEL")

label_counts = df[LABEL_COL].value_counts().reset_index()
label_counts.columns = ["Label", "Count"]
label_counts["Percentage (%)"] = (label_counts["Count"] / len(df) * 100).round(2)
print(label_counts.to_string(index=False))

# Cek imbalance
max_pct = label_counts["Percentage (%)"].max()
min_pct = label_counts["Percentage (%)"].min()
ratio   = max_pct / min_pct if min_pct > 0 else float("inf")
print(f"\n   Imbalance ratio (max/min) : {ratio:.2f}x")
if ratio > 3:
    print("   [!] Dataset cukup IMBALANCED -- pertimbangkan teknik resampling.")
else:
    print("   [OK] Dataset relatif BALANCED.")

# ── Visualisasi Countplot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
order = label_counts["Label"].tolist()
sns.countplot(data=df, x=LABEL_COL, order=order, palette="Set2", ax=ax)
ax.set_title("Distribusi Label", **FIG_TITLE_STYLE)
ax.set_xlabel("Label", fontsize=12)
ax.set_ylabel("Jumlah Data", fontsize=12)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}\n({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_label_distribution.png", dpi=150)
plt.show()
print("   [Plot] Disimpan -> plot_label_distribution.png")


# =============================================================================
# 5. ANALISIS PANJANG TEKS
# =============================================================================
section("5. ANALISIS PANJANG TEKS")

# Tambahkan fitur turunan
df["text_length"]  = df[TEXT_COL].astype(str).str.len()
df["word_count"]   = df[TEXT_COL].astype(str).str.split().str.len()

print("\n── Statistik Panjang Karakter ──")
print(df["text_length"].describe().rename({
    "count": "Jumlah", "mean": "Rata-rata", "std": "Std Dev",
    "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"
}).to_string())

print("\n── Statistik Jumlah Kata ──")
print(df["word_count"].describe().rename({
    "count": "Jumlah", "mean": "Rata-rata", "std": "Std Dev",
    "min": "Min", "25%": "Q1", "50%": "Median", "75%": "Q3", "max": "Max"
}).to_string())

# ── Visualisasi: Histogram + KDE Panjang Teks ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram panjang karakter
sns.histplot(df["text_length"], bins=40, kde=True, color=PALETTE[0], ax=axes[0])
axes[0].axvline(df["text_length"].mean(),   color="red",    linestyle="--", label=f'Mean={df["text_length"].mean():.0f}')
axes[0].axvline(df["text_length"].median(), color="orange", linestyle="--", label=f'Median={df["text_length"].median():.0f}')
axes[0].set_title("Distribusi Panjang Teks (Karakter)", **FIG_TITLE_STYLE)
axes[0].set_xlabel("Jumlah Karakter")
axes[0].set_ylabel("Frekuensi")
axes[0].legend()

# Histogram jumlah kata
sns.histplot(df["word_count"], bins=40, kde=True, color=PALETTE[1], ax=axes[1])
axes[1].axvline(df["word_count"].mean(),   color="red",    linestyle="--", label=f'Mean={df["word_count"].mean():.0f}')
axes[1].axvline(df["word_count"].median(), color="orange", linestyle="--", label=f'Median={df["word_count"].median():.0f}')
axes[1].set_title("Distribusi Jumlah Kata per Teks", **FIG_TITLE_STYLE)
axes[1].set_xlabel("Jumlah Kata")
axes[1].set_ylabel("Frekuensi")
axes[1].legend()

plt.tight_layout()
plt.savefig("plot_text_length_hist.png", dpi=150)
plt.show()
print("   [Plot] Disimpan -> plot_text_length_hist.png")

# ── Visualisasi: Boxplot Panjang Teks per Label ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=df, x=LABEL_COL, y="text_length", palette="Set2", order=order, ax=axes[0])
axes[0].set_title("Boxplot Panjang Teks per Label", **FIG_TITLE_STYLE)
axes[0].set_xlabel("Label")
axes[0].set_ylabel("Panjang Teks (Karakter)")

sns.boxplot(data=df, x=LABEL_COL, y="word_count", palette="Set2", order=order, ax=axes[1])
axes[1].set_title("Boxplot Jumlah Kata per Label", **FIG_TITLE_STYLE)
axes[1].set_xlabel("Label")
axes[1].set_ylabel("Jumlah Kata")

plt.tight_layout()
plt.savefig("plot_boxplot_per_label.png", dpi=150)
plt.show()
print("   [Plot] Disimpan -> plot_boxplot_per_label.png")


# =============================================================================
# 6. TAMBAHAN
# =============================================================================
section("6A. 10 CONTOH DATA ACAK")
print(df[[TEXT_COL, LABEL_COL]].sample(10, random_state=42).to_string(index=True))

# ── Top-20 Kata Paling Sering Muncul ─────────────────────────────────────────
section("6B. TOP 20 KATA PALING SERING MUNCUL (GLOBAL)")

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","it","its","this","that","was","are","be","been","have","has","had",
    "do","did","will","would","could","should","may","might","shall","can",
    "not","no","so","if","as","by","from","than","then","into","over","more",
    "also","just","i","you","he","she","we","they","me","him","her","us","them",
    "my","your","his","our","their","what","which","who","how","when","where",
    "rt","amp","s","t","re","ve","ll","d","m"
}

all_words = []
for text in df[TEXT_COL].astype(str):
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    all_words.extend([w for w in tokens if w not in STOPWORDS])

top20 = Counter(all_words).most_common(20)
top20_df = pd.DataFrame(top20, columns=["Kata", "Frekuensi"])
print(top20_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top20_df, x="Frekuensi", y="Kata", palette="viridis", ax=ax)
ax.set_title("Top 20 Kata Paling Sering Muncul", **FIG_TITLE_STYLE)
ax.set_xlabel("Frekuensi")
ax.set_ylabel("Kata")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("plot_top20_words.png", dpi=150)
plt.show()
print("   [Plot] Disimpan -> plot_top20_words.png")

# ── Top-20 Kata per Label ──────────────────────────────────────────────────────
section("6C. TOP 10 KATA PER LABEL")

unique_labels = df[LABEL_COL].unique()
n_labels = len(unique_labels)
fig, axes = plt.subplots(1, n_labels, figsize=(7 * n_labels, 6), squeeze=False)

for idx, lbl in enumerate(unique_labels):
    subset = df[df[LABEL_COL] == lbl][TEXT_COL].astype(str)
    words  = []
    for text in subset:
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    top10 = Counter(words).most_common(10)
    top10_df = pd.DataFrame(top10, columns=["Kata", "Frekuensi"])
    ax = axes[0][idx]
    sns.barplot(data=top10_df, x="Frekuensi", y="Kata", palette="Set2", ax=ax)
    ax.set_title(f"Top 10 Kata — Label: {lbl}", **FIG_TITLE_STYLE)
    ax.set_xlabel("Frekuensi")
    ax.set_ylabel("Kata")

plt.tight_layout()
plt.savefig("plot_top10_words_per_label.png", dpi=150)
plt.show()
print("   [Plot] Disimpan -> plot_top10_words_per_label.png")


# =============================================================================
# RINGKASAN AKHIR
# =============================================================================
section("RINGKASAN DATA UNDERSTANDING")
print(f"   Total data          : {len(df):,}")
print(f"   Jumlah kolom        : {df.shape[1]}")
print(f"   Jumlah label unik   : {df[LABEL_COL].nunique()}")
print(f"   Missing values      : {df.isnull().sum().sum():,}")
print(f"   Data duplikat       : {n_dup:,}")
print(f"   Teks kosong         : {n_empty:,}")
print(f"   Rata-rata panjang   : {df['text_length'].mean():.1f} karakter")
print(f"   Rata-rata jumlah kata: {df['word_count'].mean():.1f} kata")
print(f"   Imbalance ratio     : {ratio:.2f}x")
print(f"\n   Plot yang dihasilkan:")
print(f"     - plot_label_distribution.png")
print(f"     - plot_text_length_hist.png")
print(f"     - plot_boxplot_per_label.png")
print(f"     - plot_top20_words.png")
print(f"     - plot_top10_words_per_label.png")
print(DIVIDER)
