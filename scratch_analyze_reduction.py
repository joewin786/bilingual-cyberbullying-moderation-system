import pandas as pd
import sys

# Set standard output to handle utf-8 safely
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("dataset/dataset_fn_fp_reduction.csv")
print("=== INFO DATASET ===")
print(f"Total baris: {len(df)}")
print(f"Kolom: {df.columns.tolist()}")
print(f"Missing values: {df.isna().sum().to_dict()}")

# Distribusi label
print(f"Distribusi label (raw): {df['label'].value_counts().to_dict()}")

# Distribusi category
print(f"Distribusi category: {df['category'].value_counts().to_dict()}")

# Contoh data berdasarkan label
print("\n=== CONTOH DATA (10 random) ===")
sample_df = df.sample(10, random_state=42)[["text", "label", "category"]]
for idx, row in sample_df.iterrows():
    print(f"- Text: {row['text']}\n  Label: {row['label']} | Category: {row['category']}\n")

# Cek duplikasi teks
dup = df["text"].duplicated().sum()
print(f"\nDuplikasi teks di dalam dataset: {dup}")
