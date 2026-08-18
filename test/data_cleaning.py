# =============================================================================
# DATA CLEANING - combined_dataset.csv
# =============================================================================
# Mengatasi semua masalah yang ditemukan pada dataset:
#   1. Label tidak konsisten (6 label → 2 label binary)
#   2. Duplikat konten (same text, different label name)
#   3. Kolom redundan (Unnamed: 0, String, encoded_label)
#   4. Outlier teks terlalu panjang (opsional)
# Output: dataset_clean.csv
# =============================================================================

import pandas as pd

# ── Konfigurasi ───────────────────────────────────────────────────────────────
INPUT_PATH      = "./dataset/combined_dataset.csv"
OUTPUT_PATH     = "./dataset/dataset_clean.csv"
TEXT_COL        = "clean_text"
LABEL_COL       = "Label"
MAX_TEXT_LEN    = 500       # Set None untuk skip filter panjang teks
DIVIDER = "=" * 60

def section(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

# =============================================================================
# LOAD
# =============================================================================
section("LOAD DATASET ASLI")
df = pd.read_csv(INPUT_PATH)
print(f"  Shape awal  : {df.shape[0]:,} baris x {df.shape[1]} kolom")
print(f"  Kolom       : {df.columns.tolist()}")
print(f"\n  Distribusi label SEBELUM cleaning:")
print(df[LABEL_COL].value_counts().to_string())

# =============================================================================
# LANGKAH 1: Normalisasi Label → 2 Kelas Binary
# =============================================================================
section("LANGKAH 1: Normalisasi Label")

LABEL_MAP = {
    'negatif':      'bully',
    'negative':     'bully',
    'Bullying':     'bully',
    'positif':      'non-bully',
    'positive':     'non-bully',
    'Non-bullying': 'non-bully',
}

df['label_clean'] = df[LABEL_COL].map(LABEL_MAP)

# Cek apakah ada label yang tidak terpetakan
unmapped = df['label_clean'].isna().sum()
if unmapped > 0:
    print(f"  [!] {unmapped} baris dengan label tidak dikenal:")
    print(df[df['label_clean'].isna()][[TEXT_COL, LABEL_COL]])
    df = df.dropna(subset=['label_clean'])
    print(f"  Baris tersebut dihapus.")
else:
    print("  [OK] Semua label berhasil dipetakan.")

print(f"\n  Distribusi label SETELAH normalisasi:")
print(df['label_clean'].value_counts().to_string())

# =============================================================================
# LANGKAH 2: Hapus Duplikat Konten
# =============================================================================
section("LANGKAH 2: Hapus Duplikat Teks")

n_before = len(df)
df = df.drop_duplicates(subset=[TEXT_COL], keep='first')
n_after = len(df)
n_removed = n_before - n_after

print(f"  Baris sebelum : {n_before:,}")
print(f"  Baris setelah : {n_after:,}")
print(f"  Dihapus       : {n_removed:,} baris duplikat")

print(f"\n  Distribusi label setelah hapus duplikat:")
print(df['label_clean'].value_counts().to_string())

# =============================================================================
# LANGKAH 3: Filter Outlier Panjang Teks (Opsional)
# =============================================================================
section("LANGKAH 3: Filter Outlier Panjang Teks")

df['text_len'] = df[TEXT_COL].astype(str).str.len()
print(f"  Statistik panjang teks:")
print(df['text_len'].describe().round(1).to_string())

if MAX_TEXT_LEN:
    n_before = len(df)
    df = df[df['text_len'] <= MAX_TEXT_LEN]
    print(f"\n  Teks > {MAX_TEXT_LEN} karakter dihapus: {n_before - len(df):,} baris")
else:
    print("\n  [SKIP] Filter panjang teks tidak diaktifkan.")

# =============================================================================
# LANGKAH 4: Hapus Kolom Redundan & Rapikan
# =============================================================================
section("LANGKAH 4: Rapikan Kolom")

# Encode label numerik
label_to_int = {'bully': 1, 'non-bully': 0}
df['label_encoded'] = df['label_clean'].map(label_to_int)

# Pilih hanya kolom yang diperlukan
df_clean = df[[TEXT_COL, 'label_clean', 'label_encoded']].copy()

# Rename agar lebih standar
df_clean = df_clean.rename(columns={
    TEXT_COL:      'text',
    'label_clean': 'label',
})

# Reset index
df_clean = df_clean.reset_index(drop=True)

print(f"  Kolom hasil  : {df_clean.columns.tolist()}")
print(f"  Shape bersih : {df_clean.shape[0]:,} baris x {df_clean.shape[1]} kolom")

# =============================================================================
# RINGKASAN & SIMPAN
# =============================================================================
section("RINGKASAN CLEANING")

original_count = pd.read_csv(INPUT_PATH).shape[0]
final_count    = len(df_clean)

print(f"  Data awal          : {original_count:,} baris")
print(f"  Data setelah clean : {final_count:,} baris")
print(f"  Total dihapus      : {original_count - final_count:,} baris")
print(f"\n  Distribusi label final:")

vc = df_clean['label'].value_counts()
for lbl, cnt in vc.items():
    pct = cnt / len(df_clean) * 100
    enc = label_to_int[lbl]
    print(f"    [{enc}] {lbl:<12} : {cnt:,} ({pct:.1f}%)")

imbalance = vc.max() / vc.min()
print(f"\n  Imbalance ratio    : {imbalance:.2f}x", end=" ")
print("(BALANCED)" if imbalance <= 1.5 else "(perlu oversampling/undersampling)")

print(f"\n  Preview 5 data pertama:")
print(df_clean.head().to_string())

# Simpan
df_clean.to_csv(OUTPUT_PATH, index=False)
print(f"\n  [OK] Dataset bersih disimpan ke: '{OUTPUT_PATH}'")
print(DIVIDER)
