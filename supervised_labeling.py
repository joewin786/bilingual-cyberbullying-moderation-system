import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path so we can import src
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.models.predict import CyberbullyingPredictor

def main():
    csv_path = project_root / "data" / "processed" / "youtube_scraped.csv"
    if not csv_path.exists():
        print(f"[Error] File tidak ditemukan di {csv_path}")
        return

    print("=" * 60)
    print("SUPERVISED LABELING USING XLM-RoBERTa MODEL")
    print("=" * 60)

    # 1. Load data
    print(f"[*] Membaca data dari {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    print(f"  -> Total data: {len(df)} baris.")

    # 2. Initialize Predictor
    print("[*] Menginisialisasi XLM-RoBERTa Predictor...")
    try:
        predictor = CyberbullyingPredictor()
    except Exception as e:
        print(f"[Error] Gagal memuat model: {e}")
        return

    # 3. Predict in batches
    print("[*] Melakukan prediksi batch...")
    texts = df["text"].fillna("").tolist()
    
    # Run predictions in batches of 64
    batch_size = 64
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_results = predictor.predict_batch(batch_texts)
            predictions.extend([res.label_id for res in batch_results])
            print(f"  -> Diproses: {min(i + batch_size, len(texts))}/{len(texts)}")
        except Exception as e:
            print(f"  -> [Error] Gagal memproses batch {i}-{i+batch_size}: {e}")
            # Fallback to zeros in case of fail
            predictions.extend([0] * len(batch_texts))

    # 4. Save results back
    df["label"] = predictions
    df.to_csv(csv_path, index=False)
    
    # 5. Print statistics
    bully_count = sum(predictions)
    non_bully_count = len(predictions) - bully_count
    
    print("\n" + "=" * 60)
    print("PROSES PELABELAN SELESAI!")
    print(f"Total Labeled Bully (1)    : {bully_count}")
    print(f"Total Labeled Non-Bully (0): {non_bully_count}")
    print(f"File berhasil diupdate di   : {csv_path.absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
