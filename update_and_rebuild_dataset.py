import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    backup_dir = processed_dir / "backup_id_only"
    scraped_path = processed_dir / "youtube_scraped.csv"

    print("=" * 60)
    # Step 1: Merge scraped data into Indonesian datasets
    print("STEP 1: Menggabungkan data scraped dengan ID asli...")
    
    id_dfs = []
    # Try loading from backup first, fallback to processed
    for name in ["train.csv", "val.csv", "test.csv"]:
        backup_file = backup_dir / name
        processed_file = processed_dir / name
        if backup_file.exists():
            id_dfs.append(pd.read_csv(backup_file))
        elif processed_file.exists():
            id_dfs.append(pd.read_csv(processed_file))
            
    if not id_dfs:
        print("[Error] Tidak ada dataset Indonesia (train/val/test) yang ditemukan!")
        return

    id_all = pd.concat(id_dfs, ignore_index=True)
    print(f"  -> Total data Indonesia awal: {len(id_all)} baris.")

    # Load scraped
    if not scraped_path.exists():
        print(f"[Error] File youtube_scraped.csv tidak ditemukan di {scraped_path}")
        return
        
    scraped_df = pd.read_csv(scraped_path)
    # Keep only valid text/label columns
    scraped_df = scraped_df[["text", "label"]].dropna()
    scraped_df["label"] = scraped_df["label"].astype(int)
    print(f"  -> Total data hasil scraping baru: {len(scraped_df)} baris.")

    # Combine
    combined_id = pd.concat([id_all, scraped_df], ignore_index=True)
    combined_id.drop_duplicates(subset=["text"], keep="first", inplace=True)
    print(f"  -> Total data Indonesia gabungan (setelah dedup): {len(combined_id)} baris.")

    # Step 2: Stratified Split 80:10:10
    print("\nSTEP 2: Melakukan Stratified Split 80:10:10...")
    # Split 80% train, 20% temp (val+test)
    train_df, temp_df = train_test_split(
        combined_id,
        test_size=0.20,
        random_state=42,
        stratify=combined_id["label"]
    )
    # Split temp 50% val, 50% test (resulting in 10% val, 10% test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    print(f"  -> Hasil split:")
    print(f"     Train (80%): {len(train_df)} baris")
    print(f"     Val (10%)  : {len(val_df)} baris")
    print(f"     Test (10%) : {len(test_df)} baris")

    # Step 3: Save to backup_id_only (updating the backup splits)
    print("\nSTEP 3: Menyimpan dataset Indonesia baru ke backup...")
    backup_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(backup_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(backup_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(backup_dir / "test.csv", index=False, encoding="utf-8")
    print("  -> Backup ID asli berhasil diupdate.")

    # Save to processed directory temporarily as well, so split_en_dataset reads new sizes
    train_df.to_csv(processed_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(processed_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(processed_dir / "test.csv", index=False, encoding="utf-8")

    # Step 4: Run split_en_dataset.py
    print("\nSTEP 4: Menjalankan split_en_dataset.py untuk memilah subset EN...")
    result_en = subprocess.run([sys.executable, str(PROJECT_ROOT / "split_en_dataset.py")], capture_output=True, text=True)
    print(result_en.stdout)
    if result_en.returncode != 0:
        print(f"[Error] Gagal menjalankan split_en_dataset.py:\n{result_en.stderr}")
        return

    # Step 5: Run merge_id_en_dataset.py
    print("\nSTEP 5: Menjalankan merge_id_en_dataset.py...")
    result_merge = subprocess.run([sys.executable, str(PROJECT_ROOT / "merge_id_en_dataset.py")], capture_output=True, text=True)
    print(result_merge.stdout)
    if result_merge.returncode != 0:
        print(f"[Error] Gagal menjalankan merge_id_en_dataset.py:\n{result_merge.stderr}")
        return

    print("=" * 60)
    print("DATASET BERHASIL DIUPDATE & DIREBUILD DENGAN RASIO 80:10:10!")
    print("=" * 60)

if __name__ == "__main__":
    main()
