"""
merge_id_en_dataset.py
======================
Menggabungkan dataset ID dan EN subset menjadi file training final.
Backup file ID asli sebelum overwrite.
"""

import shutil
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

# ID files (asli)
ID_TRAIN = PROCESSED_DIR / "train.csv"
ID_VAL   = PROCESSED_DIR / "val.csv"
ID_TEST  = PROCESSED_DIR / "test.csv"

# EN subset files
EN_TRAIN = PROCESSED_DIR / "en_train_subset.csv"
EN_VAL   = PROCESSED_DIR / "en_val_subset.csv"
EN_TEST  = PROCESSED_DIR / "en_test_subset.csv"

# Backup dir
BACKUP_DIR = PROCESSED_DIR / "backup_id_only"

RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("Merge Dataset ID + EN Subset")
    print("=" * 60)

    # ── 1. Backup file ID asli ─────────────────────────────────
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for src in [ID_TRAIN, ID_VAL, ID_TEST]:
        dst = BACKUP_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  Backup: {src.name} -> {dst}")
        else:
            print(f"  Backup sudah ada: {dst}")

    # ── 2. Load ID ─────────────────────────────────────────────
    print(f"\nLoading ID datasets...")
    # Load from backup to ensure we always use the original ID files
    id_train = pd.read_csv(BACKUP_DIR / "train.csv")
    id_val   = pd.read_csv(BACKUP_DIR / "val.csv")
    id_test  = pd.read_csv(BACKUP_DIR / "test.csv")
    print(f"  ID train: {len(id_train)} | val: {len(id_val)} | test: {len(id_test)}")

    # ── 3. Load EN subset ──────────────────────────────────────
    print(f"Loading EN subsets...")
    en_train = pd.read_csv(EN_TRAIN)
    en_val   = pd.read_csv(EN_VAL)
    en_test  = pd.read_csv(EN_TEST)
    print(f"  EN train: {len(en_train)} | val: {len(en_val)} | test: {len(en_test)}")

    # ── 4. Merge ───────────────────────────────────────────────
    # Pastikan kolom konsisten (text, label saja)
    cols = ["text", "label"]
    merged_train = pd.concat([id_train[cols], en_train[cols]], ignore_index=True)
    merged_val   = pd.concat([id_val[cols],   en_val[cols]],   ignore_index=True)
    merged_test  = pd.concat([id_test[cols],  en_test[cols]],  ignore_index=True)

    # Shuffle
    merged_train = merged_train.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    merged_val   = merged_val.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    merged_test  = merged_test.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # ── 5. Save (overwrite) ────────────────────────────────────
    merged_train.to_csv(ID_TRAIN, index=False, encoding="utf-8")
    merged_val.to_csv(ID_VAL,     index=False, encoding="utf-8")
    merged_test.to_csv(ID_TEST,   index=False, encoding="utf-8")

    # ── 6. Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RINGKASAN DATASET FINAL (ID + EN)")
    print(f"{'='*60}")
    for name, df_id, df_en, df_merged in [
        ("train", id_train, en_train, merged_train),
        ("val",   id_val,   en_val,   merged_val),
        ("test",  id_test,  en_test,  merged_test),
    ]:
        dist = df_merged["label"].value_counts().to_dict()
        print(f"\n  {name}.csv:")
        print(f"    ID: {len(df_id)} + EN: {len(df_en)} = Total: {len(df_merged)}")
        print(f"    Bully (1)   : {dist.get(1, 0)}")
        print(f"    Non-bully(0): {dist.get(0, 0)}")

    total_all = len(merged_train) + len(merged_val) + len(merged_test)
    print(f"\n  Grand Total: {total_all} sampel")
    print(f"  Backup ID asli di: {BACKUP_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
