"""
prepare_dataset.py
==================
Menggabungkan dan membersihkan dataset cyberbullying dari beberapa sumber,
lalu membagi menjadi train/val/test dan menyimpannya ke data/processed/.

Sumber dataset:
- dataset/dataset_clean.csv       (label: bully / non-bully)
- dataset/combined_dataset.csv    (label: Bullying/Non-bullying, positif/negatif)
- dataset/indo/Dataset komentar Tiktok.csv      (label: 0=bully, 1=non-bully)
- dataset/indo/Dataset komentar Tiktok (1).csv   (sep=';', label: 0=bully, 1=non-bully)
- dataset/indo/dataset_relabel.csv               (new_label: 0=bully, 1=non-bully)
- dataset/indo/dataset_relabel (1).csv            (sep=';', new_label: 0=bully, 1=non-bully)

Output:
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv
- data/processed/dataset_stats.txt

Label convention (unified):
    1 = bully
    0 = non-bully
"""

import os
import sys
import yaml
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "training_config.yaml"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────
# Label Normalization
# ──────────────────────────────────────────────────────────────
def normalize_label(raw_label) -> int:
    """
    Normalkan label dari berbagai format ke:
      1 = bully
      0 = non-bully
    Return -1 jika tidak dikenali.
    """
    if pd.isna(raw_label):
        return -1

    normalized = str(raw_label).strip().lower()

    if normalized in {"bully", "bullying", "negatif", "negative"}:
        return 1
    if normalized in {"non-bully", "non-bullying", "non_bully", "non_bullying", "positif", "positive"}:
        return 0

    # Coba parse numerik (0 atau 1)
    try:
        val = int(float(normalized))
        if val in (0, 1):
            return val
    except (ValueError, TypeError):
        pass

    return -1


# ──────────────────────────────────────────────────────────────
# Dataset Loaders — Original Sources
# ──────────────────────────────────────────────────────────────
def load_dataset_clean(path: Path) -> pd.DataFrame:
    """
    Muat dataset_clean.csv
    Kolom: text, label, label_encoded
    label: 'bully' | 'non-bully'
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")

    df["label_binary"] = df["label"].apply(normalize_label)
    df = df[["text", "label_binary"]].rename(columns={"label_binary": "label"})
    df["source"] = "dataset_clean"
    return df


def load_combined_dataset(path: Path) -> pd.DataFrame:
    """
    Muat combined_dataset.csv
    Kolom: ,Label,clean_text,String,encoded_label
    Label: 'Bullying' | 'Non-bullying' | 'positif' | 'negatif'
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")

    text_col  = "clean_text" if "clean_text" in df.columns else "text"
    label_col = "Label" if "Label" in df.columns else "label"

    df["label_binary"] = df[label_col].apply(normalize_label)
    df = df[[text_col, "label_binary"]].rename(
        columns={text_col: "text", "label_binary": "label"}
    )
    df["source"] = "combined_dataset"
    return df


# ──────────────────────────────────────────────────────────────
# Dataset Loaders — New Indo Sources
# ──────────────────────────────────────────────────────────────
def load_tiktok_csv(path: Path) -> pd.DataFrame:
    """
    Muat 'Dataset komentar Tiktok.csv' (comma-separated).
    Kolom: No, komentar, label, link vid, ...
    Label: 0 = cyber bullying, 1 = non-cyber bullying
    → Flip: 0 → 1 (bully), 1 → 0 (non-bully)
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")

    df = df[["komentar", "label"]].copy()
    df = df.rename(columns={"komentar": "text"})

    # Flip labels: 0 (bully in TikTok) → 1 (bully in our system)
    df["label"] = df["label"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else -1))
    df["source"] = "tiktok"
    return df


def load_tiktok_semicolon(path: Path) -> pd.DataFrame:
    """
    Muat 'Dataset komentar Tiktok (1).csv' (semicolon-separated).
    Format: No;komentar;label;e
    Label: 0 = cyber bullying, 1 = non-cyber bullying
    → Flip sama seperti load_tiktok_csv.
    """
    logger.info(f"Membaca {path.name} (separator=';') ...")
    df = pd.read_csv(path, sep=";", encoding="utf-8", on_bad_lines="skip")
    logger.info(f"  Raw shape: {df.shape}")

    # Handle column names
    cols = df.columns.tolist()
    if len(cols) >= 3:
        # Rename first 3 columns
        col_map = {cols[0]: "no", cols[1]: "komentar", cols[2]: "label"}
        df = df.rename(columns=col_map)
    else:
        logger.warning(f"  Unexpected columns: {cols}")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = df[["komentar", "label"]].copy()
    df = df.rename(columns={"komentar": "text"})

    # Convert label to int
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Flip labels
    df["label"] = df["label"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else -1))
    df["source"] = "tiktok_v2"
    return df


def load_relabel_csv(path: Path) -> pd.DataFrame:
    """
    Muat 'dataset_relabel.csv' (comma-separated).
    Kolom: Column1, Label, clean_text, String, encoded_label, new_label
    new_label: 0 = Bullying, 1 = Non-bullying
    → Flip: 0 → 1 (bully), 1 → 0 (non-bully)
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")

    text_col = "clean_text" if "clean_text" in df.columns else "text"
    label_col = "new_label" if "new_label" in df.columns else "label"

    df = df[[text_col, label_col]].copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Flip: new_label 0=bully→1, 1=non-bully→0
    df["label"] = df["label"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else -1))
    df["source"] = "relabel"
    return df


def load_relabel_semicolon(path: Path) -> pd.DataFrame:
    """
    Muat 'dataset_relabel (1).csv' (semicolon-separated).
    Kolom: Column1;Label;clean_text;String;new_label
    new_label: 0 = Bullying, 1 = Non-bullying
    → Flip sama.
    """
    logger.info(f"Membaca {path.name} (separator=';') ...")
    df = pd.read_csv(path, sep=";", encoding="utf-8", on_bad_lines="skip")
    logger.info(f"  Raw shape: {df.shape}")

    cols = df.columns.tolist()
    if "clean_text" in cols and "new_label" in cols:
        text_col = "clean_text"
        label_col = "new_label"
    elif len(cols) >= 5:
        col_map = {cols[0]: "id", cols[1]: "label_str", cols[2]: "clean_text",
                   cols[3]: "string", cols[4]: "new_label"}
        df = df.rename(columns=col_map)
        text_col = "clean_text"
        label_col = "new_label"
    else:
        logger.warning(f"  Unexpected columns: {cols}")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = df[[text_col, label_col]].copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["label"] = df["label"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else -1))
    df["source"] = "relabel_v2"
    return df


# ──────────────────────────────────────────────────────────────
# Dataset Loaders — Augmentation (FN / FP / Codemixed)
# ──────────────────────────────────────────────────────────────
def load_fn_hatespeech(path: Path, max_samples: int = None) -> pd.DataFrame:
    """
    Muat dataset hate speech Indonesia (Ibrohim & Budi, 2019).
    Kolom: Tweet, HS, Abusive, HS_Individual, HS_Group, ...

    Mapping:
      HS=1 atau Abusive=1 → label=1 (bully)
      Hanya mengambil sampel bully untuk mengurangi False Negative.

    Args:
        path: Path ke data.csv
        max_samples: Batas maksimal sampel (None = semua ~7K bully)
    """
    logger.info(f"Membaca {path.name} (encoding=latin-1) ...")
    df = pd.read_csv(path, encoding="latin-1")
    logger.info(f"  Raw shape: {df.shape}")

    # Bully = Hate Speech ATAU Abusive
    df["label"] = ((df["HS"] == 1) | (df["Abusive"] == 1)).astype(int)

    # Hanya ambil yang bully (tujuan: mengurangi FN)
    df = df[df["label"] == 1].copy()
    logger.info(f"  Sampel bully (HS=1 | Abusive=1): {len(df)}")

    if max_samples and len(df) > max_samples:
        # Prioritaskan HS_Weak & HS_Moderate (pola FN yang sering miss)
        weak_mod_mask = pd.Series(False, index=df.index)
        if "HS_Weak" in df.columns:
            weak_mod_mask = weak_mod_mask | (df["HS_Weak"] == 1)
        if "HS_Moderate" in df.columns:
            weak_mod_mask = weak_mod_mask | (df["HS_Moderate"] == 1)

        priority = df[weak_mod_mask]
        rest = df[~weak_mod_mask]

        if len(priority) >= max_samples:
            df = priority.sample(n=max_samples, random_state=42)
        else:
            need = max_samples - len(priority)
            df = pd.concat([
                priority,
                rest.sample(n=min(need, len(rest)), random_state=42)
            ])
        logger.info(f"  Dibatasi ke {len(df)} sampel (max_samples={max_samples})")

    df = df[["Tweet", "label"]].rename(columns={"Tweet": "text"})
    df["source"] = "fn_hatespeech_id"
    return df


def load_fp_augmentation(path: Path) -> pd.DataFrame:
    """
    Muat dataset augmentasi FP (hard-negatives) buatan manual.
    Format CSV: text, label, category

    Semua data dimuat apa adanya (label dari file).
    Otomatis handle format dimana seluruh baris terbungkus dalam quote.
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info(f"  Raw shape: {df.shape}")

    # Handle format dimana semua kolom terbungkus jadi 1 kolom
    # (misalnya: "text,label,category" sebagai satu field)
    if len(df.columns) == 1 and "," in df.columns[0]:
        logger.info("  ℹ️ Format terdeteksi: kolom terbungkus dalam quote, re-parsing ...")
        raw_text = path.read_text(encoding="utf-8-sig")
        lines = raw_text.strip().split("\n")

        # Header
        header = lines[0].strip('"').strip()

        # Parse baris data: split dari kanan karena text bisa mengandung koma
        rows = []
        for i, line in enumerate(lines[1:], start=2):
            line = line.strip().strip('"')
            if not line:
                continue
            # Split dari kanan: ...,label,category → max 2 split dari kanan
            parts = line.rsplit(",", 2)
            if len(parts) == 3:
                rows.append({"text": parts[0], "label": parts[1], "category": parts[2]})
            else:
                logger.warning(f"  ⚠️ Baris {i} tidak valid, dilewati: {line[:80]}")

        df = pd.DataFrame(rows)
        # Konversi label ke numerik
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        logger.info(f"  Re-parsed shape: {df.shape}")

    if "text" not in df.columns or "label" not in df.columns:
        logger.warning(f"  ⚠️ Kolom 'text' atau 'label' tidak ditemukan: {list(df.columns)}")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = df[["text", "label"]].copy()
    df["source"] = "manual_augmentation"
    return df


def load_codemixed_samples(path: Path) -> pd.DataFrame:
    """
    Muat dataset code-mixed (campuran Indonesia-Inggris).
    Format CSV: text, label
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")

    df = df[["text", "label"]].copy()
    df["source"] = "codemixed"
    return df


def load_contrastive_dataset(path: Path) -> pd.DataFrame:
    """
    Muat contrastive_cyberbullying_starter_dataset.csv.
    Kolom: text, label
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")
    df = df[["text", "label"]].copy()
    df["source"] = "contrastive_dataset"
    return df


def load_cyberbullying_1000(path: Path) -> pd.DataFrame:
    """
    Muat cyberbullying_dataset_1000.csv.
    Kolom: text, label (numerik 0/1 atau string)
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].apply(normalize_label)
    df["source"] = "cyberbullying_1000"
    return df


def load_fn_fp_reduction(path: Path) -> pd.DataFrame:
    """
    Muat dataset_fn_fp_reduction.csv.
    Kolom: id, text, label (bullying / non_bullying), category (FN_target / FP_target)
    Dataset khusus dirancang untuk mengurangi False Negatives (sarkasme/sindiran halus)
    dan False Positives (kata kasar dalam konteks candaan/netral).
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].apply(normalize_label)
    df["source"] = "fn_fp_reduction_v2" if "v2" in path.name.lower() else "fn_fp_reduction"
    return df


def load_research_dataset(path: Path) -> pd.DataFrame:
    """
    Muat 'Dataset-Research.csv'.
    Kolom: sentiment, comment
    sentiment: -1 = bully, 1 = non-bully
    → Map: -1 → 1 (bully), 1 → 0 (non-bully)
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")
    df = df[["comment", "sentiment"]].copy()
    df = df.rename(columns={"comment": "text"})
    df["label"] = df["sentiment"].apply(lambda x: 1 if x == -1 else (0 if x == 1 else -1))
    df["source"] = "dataset_research"
    return df


def load_translated_dataset(path: Path) -> pd.DataFrame:
    """
    Muat 'translated_en_to_id.csv'.
    Kolom: text, label
    """
    logger.info(f"Membaca {path.name} ...")
    df = pd.read_csv(path, encoding="utf-8")
    logger.info(f"  Raw shape: {df.shape}")
    df = df[["text", "label"]].copy()
    df["source"] = "translated_en_to_id"
    return df



# ──────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus baris tidak valid, duplikat, dan teks kosong."""
    before = len(df)

    # Hapus label tidak dikenali
    df = df[df["label"].isin([0, 1])].copy()

    # Hapus teks kosong / NaN
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # Hapus duplikat berdasarkan teks (keep first)
    df = df.drop_duplicates(subset=["text"], keep="first")

    after = len(df)
    logger.info(f"  Setelah cleaning: {after} baris (dihapus {before - after})")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Statistics Reporter
# ──────────────────────────────────────────────────────────────
def print_stats(df: pd.DataFrame, split_name: str = "full"):
    total = len(df)
    bully     = (df["label"] == 1).sum()
    non_bully = (df["label"] == 0).sum()
    logger.info(
        f"  [{split_name}] total={total} | bully={bully} ({bully/total*100:.1f}%) "
        f"| non-bully={non_bully} ({non_bully/total*100:.1f}%)"
    )


def save_stats(df_full, df_train, df_val, df_test, output_path: Path):
    lines = [
        "=" * 60,
        "DATASET STATISTICS",
        "=" * 60,
        f"Total samples : {len(df_full)}",
        f"  Bully       : {(df_full['label']==1).sum()}",
        f"  Non-bully   : {(df_full['label']==0).sum()}",
        "",
        f"Train split   : {len(df_train)} ({len(df_train)/len(df_full)*100:.1f}%)",
        f"Val split     : {len(df_val)} ({len(df_val)/len(df_full)*100:.1f}%)",
        f"Test split    : {len(df_test)} ({len(df_test)/len(df_full)*100:.1f}%)",
        "",
        "Source breakdown:",
    ]
    if "source" in df_full.columns:
        for src, grp in df_full.groupby("source"):
            lines.append(
                f"  {src}: {len(grp)} | bully={(grp['label']==1).sum()} | "
                f"non-bully={(grp['label']==0).sum()}"
            )
    lines.append("=" * 60)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Statistik disimpan ke {output_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    cfg = load_config(CONFIG_PATH)
    data_cfg = cfg["data"]

    raw_dir       = PROJECT_ROOT / data_cfg["raw_dir"]
    processed_dir = PROJECT_ROOT / data_cfg["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    random_seed = data_cfg["random_seed"]
    train_ratio = data_cfg["train_ratio"]
    val_ratio   = data_cfg["val_ratio"]

    # ── 1. Load all sources ──────────────────────────────────
    # Kami menyusun urutan pemuatan (frames) berdasarkan prioritas kebersihan label (label quality hierarchy).
    # Dataset dengan label manual berkualitas tinggi/koreksi ditaruh di awal, sehingga ketika
    # drop_duplicates(keep="first") dijalankan, label yang benar/koreksi tetap dipertahankan
    # sedangkan label lama/kotor dari raw sources di bawahnya akan dibuang.
    frames = []
    aug_cfg = data_cfg.get("augmentation", {})
    indo_dir = raw_dir / "indo"

    # --- Prioritas 1: Augmentasi Manual / Koreksi Spesifik (Kualitas Tertinggi) ---
    if aug_cfg.get("enabled", False):
        logger.info("\n📦 Memuat data augmentasi prioritas tinggi ...")

        # Contrastive: Pasangan contrastive buatan manual (koreksi FP/FN)
        contrastive_path = PROJECT_ROOT / aug_cfg.get("contrastive_dataset", "dataset/contrastive_cyberbullying_starter_dataset.csv")
        if contrastive_path.exists():
            frames.append(load_contrastive_dataset(contrastive_path))
        else:
            logger.warning(f"  File contrastive dataset tidak ditemukan: {contrastive_path}")

        # Cyberbullying 1000: Dataset kontras tambahan (koreksi FP/FN)
        cb1000_path = PROJECT_ROOT / aug_cfg.get("cyberbullying_1000", "dataset/cyberbullying_dataset_1000.csv")
        if cb1000_path.exists():
            frames.append(load_cyberbullying_1000(cb1000_path))
        else:
            logger.warning(f"  File cyberbullying_dataset_1000 tidak ditemukan: {cb1000_path}")

        # FN/FP Reduction: Dataset khusus sarkasme+kata kasar konteks aman (koreksi FN & FP)
        fn_fp_path = PROJECT_ROOT / aug_cfg.get("fn_fp_reduction", "dataset/dataset_fn_fp_reduction.csv")
        if fn_fp_path.exists():
            frames.append(load_fn_fp_reduction(fn_fp_path))
        else:
            logger.warning(f"  File dataset_fn_fp_reduction tidak ditemukan: {fn_fp_path}")

        # FP: Hard-negatives buatan manual (koreksi FP)
        fp_path = PROJECT_ROOT / aug_cfg.get("fp_hard_negatives", "dataset/FP/hard_negatives.csv")
        if fp_path.exists():
            frames.append(load_fp_augmentation(fp_path))
        else:
            logger.info(f"  ℹ️ FP augmentation belum tersedia: {fp_path}")

        # Codemixed: Sampel code-mixed Indo-English
        cm_path = PROJECT_ROOT / aug_cfg.get("codemixed", "data/processed/codemixed_samples.csv")
        if cm_path.exists():
            frames.append(load_codemixed_samples(cm_path))
        else:
            logger.info(f"  ℹ️ Codemixed belum tersedia: {cm_path}")

    # --- Prioritas 2: Relabeled Datasets (Data yang sudah dikoreksi manual) ---
    path_relabel = indo_dir / "dataset_relabel.csv"
    if path_relabel.exists():
        frames.append(load_relabel_csv(path_relabel))
    else:
        logger.warning(f"File tidak ditemukan: {path_relabel}")

    path_relabel2 = indo_dir / "dataset_relabel (1).csv"
    if path_relabel2.exists():
        frames.append(load_relabel_semicolon(path_relabel2))
    else:
        logger.warning(f"File tidak ditemukan: {path_relabel2}")

    path_research = raw_dir / "Dataset-Research.csv"
    if path_research.exists():
        frames.append(load_research_dataset(path_research))
    else:
        logger.warning(f"File tidak ditemukan: {path_research}")

    # --- Prioritas 3: New Indo Sources (Tiktok) ---
    path_tiktok = indo_dir / "Dataset komentar Tiktok.csv"
    if path_tiktok.exists():
        frames.append(load_tiktok_csv(path_tiktok))
    else:
        logger.warning(f"File tidak ditemukan: {path_tiktok}")

    path_tiktok2 = indo_dir / "Dataset komentar Tiktok (1).csv"
    if path_tiktok2.exists():
        frames.append(load_tiktok_semicolon(path_tiktok2))
    else:
        logger.warning(f"File tidak ditemukan: {path_tiktok2}")

    # --- Prioritas 4: Hate Speech Indonesia (Hanya ambil bully untuk mengurangi FN) ---
    if aug_cfg.get("enabled", False):
        fn_path = PROJECT_ROOT / aug_cfg.get("fn_hatespeech", "dataset/FN/data.csv")
        if fn_path.exists():
            fn_max = aug_cfg.get("fn_max_samples", None)
            frames.append(load_fn_hatespeech(fn_path, max_samples=fn_max))
        else:
            logger.warning(f"  File tidak ditemukan: {fn_path}")

    # --- Prioritas 5: Original Raw Sources (Bisa mengandung label kotor/salah) ---
    path_clean = raw_dir / "dataset_clean.csv"
    if path_clean.exists():
        frames.append(load_dataset_clean(path_clean))
    else:
        logger.warning(f"File tidak ditemukan: {path_clean}")

    path_combined = raw_dir / "combined_dataset.csv"
    if path_combined.exists():
        frames.append(load_combined_dataset(path_combined))
    else:
        logger.warning(f"File tidak ditemukan: {path_combined}")

    if not frames:
        logger.error("Tidak ada dataset yang berhasil dimuat. Keluar.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total setelah merge: {len(df)} baris")

    # ── 2. Clean ─────────────────────────────────────────────
    logger.info("Membersihkan dataset ...")
    df = clean_dataframe(df)
    print_stats(df, "full")

    # ── 3. Split ──────────────────────────────────────────────
    logger.info("Membagi dataset (stratified) ...")

    # Train + (Val+Test)
    df_train, df_temp = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df["label"],
        random_state=random_seed,
    )

    # Val + Test (dari sisa)
    relative_val = val_ratio / (1 - train_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=1 - relative_val,
        stratify=df_temp["label"],
        random_state=random_seed,
    )

    print_stats(df_train, "train")
    print_stats(df_val,   "val")
    print_stats(df_test,  "test")

    # ── 4. Simpan ─────────────────────────────────────────────
    save_cols = ["text", "label", "source"]

    train_path = processed_dir / data_cfg["train_file"]
    val_path   = processed_dir / data_cfg["val_file"]
    test_path  = processed_dir / data_cfg["test_file"]

    df_train[save_cols].to_csv(train_path, index=False, encoding="utf-8")
    df_val[save_cols].to_csv(val_path,   index=False, encoding="utf-8")
    df_test[save_cols].to_csv(test_path, index=False, encoding="utf-8")

    logger.info(f"✅ Train disimpan → {train_path}")
    logger.info(f"✅ Val   disimpan → {val_path}")
    logger.info(f"✅ Test  disimpan → {test_path}")

    # ── 5. Statistik ──────────────────────────────────────────
    stats_path = processed_dir / "dataset_stats.txt"
    save_stats(df, df_train, df_val, df_test, stats_path)

    logger.info("✅ Persiapan dataset selesai.")


if __name__ == "__main__":
    main()
