import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Daftar kata kunci untuk mendeteksi label yang salah
# Rule 1: Kata kunci yang pasti aman (non-bully/0) tapi sering salah dilabeli sebagai bully (1)
SAFE_KEYWORDS = [
    r"\bgiveaway\b",
    r"\bpulsa\b",
    r"\bdapetin\b",
    r"\bhadiah\b",
    r"\bshopee\b",
    r"\bgopay\b",
    r"\bpromo\b",
    r"\bkuota\b",
    r"\bgratis\b",
    r"\bsemoga.*beruntung\b",
    r"\bklik.*link\b",
    r"\bspesifikasi.*ram\b",
    r"\bklik.*tautan\b",
]

# Rule 2: Kata kunci yang pasti toxic/bully (1) tapi salah dilabeli sebagai non-bully (0)
TOXIC_KEYWORDS = [
    r"\bgoblog\b",
    r"\bgoblok\b",
    r"\bdungu\b",
    r"\btolol\b",
    r"\bkontol\b",
    r"\bmemek\b",
    r"\bngentot\b",
    r"\blonte\b",
    r"\bperek\b",
    r"\bpecun\b",
    r"\bkelamin\b",
    r"\bngewe\b",
    r"\banjing\b",
    r"\banj\b",
    r"\bbangsat\b",
    r"\bbgsd\b",
    r"\bngentod\b",
    r"\bkeparat\b",
    r"\bidiot\b",
    r"\bautis\b",
    r"\bsinting\b",
    r"\bsetan\b",
    r"\biblis\b",
    r"\btai kucing\b",
    r"\btaik\b",
    r"\bsegumpal tai\b",
    r"\bnigger\b",
    r"\bniggers\b",
    r"\bnicca\b",
    r"\bfag\b",
    r"\bfags\b",
    r"\bdyke\b",
    r"\bhoodrats\b",
    r"\bghetto trash\b",
    r"\blibertaritards\b",
    r"\btrash\b",
    r"\bscum\b",
    r"\bmusnahkan.*bumi\b",
    r"\bhomo.*musnahkan\b",
    r"\bbitch\b",
    r"\bhoe\b",
    r"\basses\b",
    r"\bcooning\b",
]

def auto_correct_labels():
    for file_name in ["train.csv", "val.csv", "test.csv"]:
        csv_path = PROJECT_ROOT / "data" / "processed" / file_name
        if not csv_path.exists():
            print(f"File tidak ditemukan: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        original_labels = df["label"].copy()
        
        corrected_to_0 = 0
        corrected_to_1 = 0
        
        for idx, row in df.iterrows():
            text = str(row["text"]).lower()
            label = int(row["label"])
            
            # Rule 1: Jika dilabeli bully (1) tapi mengandung keyword aman -> ubah ke 0
            if label == 1:
                is_safe = False
                for pattern in SAFE_KEYWORDS:
                    if re.search(pattern, text):
                        is_safe = True
                        break
                if is_safe:
                    df.at[idx, "label"] = 0
                    corrected_to_0 += 1
                    
            # Rule 2: Jika dilabeli non-bully (0) tapi mengandung keyword toxic -> ubah ke 1
            elif label == 0:
                is_toxic = False
                for pattern in TOXIC_KEYWORDS:
                    if re.search(pattern, text):
                        is_toxic = True
                        break
                if is_toxic:
                    df.at[idx, "label"] = 1
                    corrected_to_1 += 1
                    
        # Simpan perubahan jika ada
        total_corrected = corrected_to_0 + corrected_to_1
        if total_corrected > 0:
            # Backup file
            backup_path = csv_path.with_name(f"{csv_path.stem}_backup_auto.csv")
            df_old = pd.read_csv(csv_path)
            df_old.to_csv(backup_path, index=False)
            
            df.to_csv(csv_path, index=False)
            print(f"=== {file_name} ===")
            print(f"Koreksi ke 0 (Non-Bully): {corrected_to_0}")
            print(f"Koreksi ke 1 (Bully)    : {corrected_to_1}")
            print(f"Total perubahan         : {total_corrected}")
            print(f"Backup disimpan ke      : {backup_path}\n")
        else:
            print(f"=== {file_name} ===")
            print("Tidak ada label yang perlu dikoreksi berdasarkan aturan.\n")

if __name__ == "__main__":
    auto_correct_labels()
