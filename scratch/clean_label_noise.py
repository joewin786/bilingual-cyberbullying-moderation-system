import pandas as pd
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Corrections mapping: text substring -> correct label
CORRECTIONS = {
    # 1. Clear bullying/hate speech that was mislabeled as non-bully (0) -> change to 1
    "you are nothing but a bad kike shill": 1,
    "jew fag that was stalking a famous fag": 1,
    "si lonte pelakor": 1,
    "s lonte bukan sensasi": 1,
    "lu diem aja cantik mel": 1,
    "lu gnteng klo gak senyum": 1,
    "ketiaknya.......": 1,
    
    # 2. Clear non-bullying/polite/normal sentences that were mislabeled as bully (1) -> change to 0
    "aku minta doyoung ajaa makasih yaa": 0,
    "mengubah ban depan myvi": 0,
    "gilaa netijen korea kalo ngomong": 0,
    "something far more dangerous than birds": 0,
    "didnt collude with russia": 0,
    "post insightful obs, crickets": 0,
}

def clean_csv_file(file_path: Path) -> int:
    """Read CSV, check if any target text exists, correct label, and save back if modified."""
    try:
        # Detect delimiter (comma or semicolon)
        # Try comma first
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print(f"  [Error] Gagal membaca {file_path.name}: {e}")
            return 0
    except Exception as e:
        print(f"  [Error] Gagal membaca {file_path.name}: {e}")
        return 0

    # Ensure columns exist
    text_col = None
    label_col = None
    
    # Try finding the text column
    for col in df.columns:
        if str(col).lower() in ["text", "comment", "komentar", "clean_text", "tweet"]:
            text_col = col
            break
            
    # Try finding the label column
    for col in df.columns:
        if str(col).lower() in ["label", "sentiment", "new_label", "label_encoded"]:
            label_col = col
            break

    if text_col is None or label_col is None:
        return 0

    modified = False
    corrections_count = 0
    
    # Unambiguous toxic words that should always be categorized as bully (1)
    TOXIC_WORDS = ["bangsat", "lonte", "kontol", "memek", "ngentot", "bajingan", "pantek", "perek", "pepek", "ngewe", "faggot", "kike"]

    for idx, row in df.iterrows():
        text_val = str(row[text_col]).lower()
        current_label = row[label_col]
        
        # 1. Exact corrections from mapping dictionary
        corrected = False
        for substring, correct_label in CORRECTIONS.items():
            if substring.lower() in text_val:
                # Cast correct_label appropriately if column is float
                target_val = type(current_label)(correct_label) if pd.notna(current_label) else correct_label
                if current_label != target_val:
                    df.at[idx, label_col] = target_val
                    modified = True
                    corrections_count += 1
                    corrected = True
                    print(f"    [{file_path.name}] Corrected (Dict): '{row[text_col][:40]}...' | {current_label} -> {target_val}")
                break
        
        if corrected:
            continue
            
        # 2. Rule-based correction for highly toxic words labeled as non-bully
        is_non_bully = False
        if pd.notna(current_label):
            lbl_str = str(current_label).strip().lower()
            if lbl_str in ["0", "0.0", "non-bully", "non-bullying", "non_bully", "non_bullying", "positif", "positive"]:
                is_non_bully = True
            elif lbl_str == "1" and "sentiment" in str(label_col).lower():  # Dataset-Research uses 1 for non-bully
                is_non_bully = True
                
        if is_non_bully:
            for word in TOXIC_WORDS:
                if word in text_val:
                    if type(current_label) == str:
                        if current_label.lower() in ["positif", "positive"]:
                            target_val = "negatif"
                        else:
                            target_val = "bully"
                    else:
                        if "sentiment" in str(label_col).lower():
                            target_val = type(current_label)(-1)  # -1 is bully in Dataset-Research
                        else:
                            target_val = type(current_label)(1)   # 1 is bully in general
                            
                    if current_label != target_val:
                        df.at[idx, label_col] = target_val
                        modified = True
                        corrections_count += 1
                        print(f"    [{file_path.name}] Corrected (Toxic Word '{word}'): '{row[text_col][:40]}...' | {current_label} -> {target_val}")
                    break

    if modified:
        try:
            df.to_csv(file_path, index=False, encoding="utf-8")
        except Exception as e:
            print(f"  [Error] Gagal menyimpan {file_path.name}: {e}")
            return 0

    return corrections_count

def main():
    print("=" * 60)
    print("MEMBERSIHKAN LABEL NOISE DI SEMUA CSV")
    print("=" * 60)

    # Directories to search
    target_dirs = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "dataset",
    ]

    total_corrections = 0
    cleaned_files = 0

    for directory in target_dirs:
        if not directory.exists():
            continue
            
        print(f"[*] Memindai direktori: {directory.relative_to(PROJECT_ROOT)}")
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv"):
                    file_path = Path(root) / file
                    count = clean_csv_file(file_path)
                    if count > 0:
                        total_corrections += count
                        cleaned_files += 1

    print("\n" + "=" * 60)
    print("PROSES PEMBERSIHAN SELESAI!")
    print(f"Total file yang dibersihkan: {cleaned_files}")
    print(f"Total baris yang dikoreksi  : {total_corrections}")
    print("=" * 60)

if __name__ == "__main__":
    main()
