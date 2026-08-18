import pandas as pd
import random

df = pd.read_csv("data/processed/train.csv")
non_bully = df[df["label"]==0]["text"].tolist()
bully = df[df["label"]==1]["text"].tolist()

print(f"Total: {len(df)} | Bully: {len(bully)} | Non-bully: {len(non_bully)}")

print(f"\nContoh non-bully (10 random):")
random.seed(42)
for t in random.sample(non_bully, min(10, len(non_bully))):
    print(f"  - {t[:80]}")

print(f"\nContoh bully (10 random):")
for t in random.sample(bully, min(10, len(bully))):
    print(f"  - {t[:80]}")

print(f"\nRata-rata panjang teks:")
print(f"  Bully:     {df[df['label']==1]['text'].str.len().mean():.0f} karakter")
print(f"  Non-bully: {df[df['label']==0]['text'].str.len().mean():.0f} karakter")

# Cek teks pendek (1-2 kata)
short = df[df["text"].str.split().str.len() <= 2]
print(f"\nTeks pendek (1-2 kata): {len(short)}")
print(f"  Bully: {len(short[short['label']==1])}")
print(f"  Non-bully: {len(short[short['label']==0])}")
