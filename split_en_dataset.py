"""
split_en_dataset.py
====================
Kurasi dan subset dataset Bahasa Inggris agar seimbang dengan dataset
Bahasa Indonesia untuk training multilingual XLM-RoBERTa.

Strategi:
  - Train: scoring-based selection (bukan random), rasio ID:EN = 1:1.75
  - Val/Test: random balanced sampling, ukuran sama dengan val/test ID (870)
  - Deduplicate text antar split untuk mencegah data leakage
"""

import re
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")

EN_TRAIN_PATH = PROCESSED_DIR / "en_train.csv"
EN_VAL_PATH   = PROCESSED_DIR / "en_val.csv"
EN_TEST_PATH  = PROCESSED_DIR / "en_test.csv"

OUT_TRAIN = PROCESSED_DIR / "en_train_subset.csv"
OUT_VAL   = PROCESSED_DIR / "en_val_subset.csv"
OUT_TEST  = PROCESSED_DIR / "en_test_subset.csv"

# ── Parameters ─────────────────────────────────────────────────
# Load dynamically from current ID-only processed datasets
try:
    ID_TRAIN_SIZE = len(pd.read_csv(PROCESSED_DIR / "train.csv"))
    ID_VAL_SIZE = len(pd.read_csv(PROCESSED_DIR / "val.csv"))
    ID_TEST_SIZE = len(pd.read_csv(PROCESSED_DIR / "test.csv"))
except Exception:
    # Fallback to defaults if files don't exist yet
    ID_TRAIN_SIZE = 6094
    ID_VAL_SIZE = 1306
    ID_TEST_SIZE = 1306

EN_TRAIN_RATIO = 1.75  # EN train = ID train * 1.75
EN_TRAIN_TARGET = int(ID_TRAIN_SIZE * EN_TRAIN_RATIO)
EN_VAL_TARGET = ID_VAL_SIZE
EN_TEST_TARGET = ID_TEST_SIZE
RANDOM_STATE = 42
IDEAL_WORD_COUNT = 10  # Panjang ideal teks cyberbullying Discord/TikTok

# Kata ganti orang kedua (indikasi serangan personal langsung)
SECOND_PERSON_WORDS = {
    "you", "your", "youre", "you're", "yours", "yourself",
    "ur", "u", "you've", "youve", "yall", "y'all",
    "you'd", "youd", "you'll", "youll",
}

def load_and_validate(path: Path, name: str) -> pd.DataFrame:
    """Load CSV dan validasi kolom."""
    print(f"Loading {name}: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    assert "text" in df.columns and "label" in df.columns, \
        f"Kolom 'text' atau 'label' tidak ditemukan di {name}!"
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df = df[df["label"].isin([0, 1])].copy()
    print(f"  -> {len(df)} baris valid, label dist: {df['label'].value_counts().to_dict()}")
    return df


def deduplicate_across_splits(df_train, df_val, df_test):
    """Hapus teks duplikat: prioritaskan val/test, buang dari train."""
    val_test_texts = set(df_val["text"].tolist()) | set(df_test["text"].tolist())
    before = len(df_train)
    df_train = df_train[~df_train["text"].isin(val_test_texts)].copy()
    removed = before - len(df_train)
    print(f"\n[Dedup] Hapus {removed} baris duplikat dari en_train (overlap dengan val/test)")

    # Dedup val vs test
    test_texts = set(df_test["text"].tolist())
    before_val = len(df_val)
    df_val = df_val[~df_val["text"].isin(test_texts)].copy()
    removed_val = before_val - len(df_val)
    if removed_val > 0:
        print(f"[Dedup] Hapus {removed_val} baris duplikat dari en_val (overlap dengan test)")

    # Internal dedup per split
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        before_d = len(df)
        df.drop_duplicates(subset=["text"], keep="first", inplace=True)
        dup = before_d - len(df)
        if dup > 0:
            print(f"[Dedup] Hapus {dup} duplikat internal dari en_{name}")

    return df_train, df_val, df_test


def compute_relevance_score(text: str) -> float:
    """
    Hitung skor relevansi teks untuk cyberbullying detection.
    
    Scoring:
    1. Word count proximity ke IDEAL_WORD_COUNT (max 1.0)
    2. Bonus untuk kata ganti orang kedua (indikasi personal attack)
    """
    words = text.lower().split()
    word_count = len(words)

    # Skor 1: Kedekatan ke ideal word count (Gaussian-like, peak di 10)
    distance = abs(word_count - IDEAL_WORD_COUNT)
    word_score = max(0, 1.0 - (distance / 20.0))  # Menurun linear, 0 di 30+ kata

    # Skor 2: Bonus kata ganti orang kedua
    # Tokenize sederhana: split + hapus tanda baca trailing
    clean_words = set(re.sub(r"[^\w']", "", w) for w in words)
    second_person_matches = clean_words & SECOND_PERSON_WORDS
    person_bonus = min(len(second_person_matches) * 0.15, 0.6)  # Max bonus 0.6

    return round(word_score + person_bonus, 4)


def select_train_subset(df: pd.DataFrame, target_total: int) -> pd.DataFrame:
    """Pilih subset train berdasarkan relevance scoring, balanced 50/50."""
    target_per_class = target_total // 2

    df = df.copy()
    df["_score"] = df["text"].apply(compute_relevance_score)

    results = []
    for label in [0, 1]:
        label_name = "non-bully" if label == 0 else "bully"
        df_label = df[df["label"] == label].copy()

        available = len(df_label)
        take = min(target_per_class, available)

        # Sort by score descending, take top-N
        df_label = df_label.sort_values("_score", ascending=False).head(take)
        results.append(df_label)
        print(f"  Train subset [{label_name}]: target={target_per_class}, "
              f"available={available}, taken={take}, "
              f"avg_score={df_label['_score'].mean():.4f}")

    df_out = pd.concat(results).sample(frac=1.0, random_state=RANDOM_STATE)
    df_out = df_out.drop(columns=["_score"])
    return df_out.reset_index(drop=True)


def select_random_subset(df: pd.DataFrame, target_total: int) -> pd.DataFrame:
    """Pilih subset random balanced 50/50."""
    target_per_class = target_total // 2
    results = []
    for label in [0, 1]:
        df_label = df[df["label"] == label]
        take = min(target_per_class, len(df_label))
        results.append(df_label.sample(n=take, random_state=RANDOM_STATE))
    return pd.concat(results).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def main():
    print("=" * 60)
    print("Split & Kurasi Dataset Bahasa Inggris")
    print("=" * 60)

    # ── 1. Load ────────────────────────────────────────────────
    df_train = load_and_validate(EN_TRAIN_PATH, "en_train")
    df_val   = load_and_validate(EN_VAL_PATH,   "en_val")
    df_test  = load_and_validate(EN_TEST_PATH,  "en_test")

    # ── 2. Deduplicate ─────────────────────────────────────────
    df_train, df_val, df_test = deduplicate_across_splits(df_train, df_val, df_test)
    print(f"\nSetelah dedup: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    # ── 3. Select subsets ──────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"Target EN train subset: {EN_TRAIN_TARGET} (rasio ID:EN = 1:{EN_TRAIN_RATIO})")
    print(f"{'-'*60}")
    train_subset = select_train_subset(df_train, EN_TRAIN_TARGET)

    print(f"\nTarget EN val subset: {EN_VAL_TARGET}")
    val_subset = select_random_subset(df_val, EN_VAL_TARGET)

    print(f"Target EN test subset: {EN_TEST_TARGET}")
    test_subset = select_random_subset(df_test, EN_TEST_TARGET)

    # ── 4. Save ────────────────────────────────────────────────
    train_subset[["text", "label"]].to_csv(OUT_TRAIN, index=False, encoding="utf-8")
    val_subset[["text", "label"]].to_csv(OUT_VAL, index=False, encoding="utf-8")
    test_subset[["text", "label"]].to_csv(OUT_TEST, index=False, encoding="utf-8")

    # ── 5. Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RINGKASAN HASIL")
    print(f"{'='*60}")
    for name, df, path in [
        ("en_train_subset", train_subset, OUT_TRAIN),
        ("en_val_subset",   val_subset,   OUT_VAL),
        ("en_test_subset",  test_subset,  OUT_TEST),
    ]:
        dist = df["label"].value_counts().to_dict()
        print(f"\n  {name}:")
        print(f"    Total baris : {len(df)}")
        print(f"    Bully (1)   : {dist.get(1, 0)}")
        print(f"    Non-bully(0): {dist.get(0, 0)}")
        print(f"    Disimpan ke : {path}")

    total_train_combined = ID_TRAIN_SIZE + len(train_subset)
    print(f"\n{'-'*60}")
    print(f"  Rasio akhir ID:EN (train):")
    print(f"    ID train     : {ID_TRAIN_SIZE}")
    print(f"    EN train sub : {len(train_subset)}")
    print(f"    Total train  : {total_train_combined}")
    print(f"    Rasio ID:EN  : 1 : {len(train_subset)/ID_TRAIN_SIZE:.2f}")
    print(f"{'='*60}")

    # ── 6. Tampilkan 5 contoh dari train subset ────────────────
    print(f"\n{'='*60}")
    print("5 CONTOH BARIS DARI en_train_subset.csv (skor tertinggi)")
    print(f"{'='*60}")
    # Re-compute score for display
    sample = train_subset.head(10).copy()
    sample["_score"] = sample["text"].apply(compute_relevance_score)
    sample = sample.sort_values("_score", ascending=False).head(5)
    for i, row in sample.iterrows():
        label_str = "bully" if row["label"] == 1 else "non-bully"
        print(f"\n  [{label_str}] (score={row['_score']:.4f})")
        print(f"  \"{row['text'][:120]}{'...' if len(row['text']) > 120 else ''}\"")


if __name__ == "__main__":
    main()
