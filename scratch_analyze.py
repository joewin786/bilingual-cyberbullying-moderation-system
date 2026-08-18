import pandas as pd

df = pd.read_csv("dataset/cyberbullying_dataset_1000.csv")
print("=== INFO DATASET ===")
print(f"Total baris: {len(df)}")
print(f"Kolom: {df.columns.tolist()}")
print(f"Missing values: {df.isna().sum().to_dict()}")

dist = df["label"].value_counts().to_dict()
print(f"Distribusi label: {dist} (0=non-bully, 1=bully)")

print("\n=== CONTOH DATA (10 random) ===")
print(df.sample(10, random_state=42)[["text", "label"]])

# Cek persimpangan/duplikat teks
dup = df["text"].duplicated().sum()
print(f"\nDuplikasi teks di dalam dataset: {dup}")
