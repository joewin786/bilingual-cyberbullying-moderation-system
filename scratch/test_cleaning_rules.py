import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"

TOXIC_WORDS = [
    "bangsat", "lonte", "kontol", "memek", "ngentot", "bajingan", 
    "pantek", "perek", "pepek", "ngewe", "faggot", "kike"
]

if train_path.exists():
    df = pd.read_csv(train_path)
    print(f"Total train rows: {len(df)}")
    
    mislabeled = []
    for idx, row in df.iterrows():
        text_lower = str(row["text"]).lower()
        if row["label"] == 0:
            for word in TOXIC_WORDS:
                if word in text_lower:
                    mislabeled.append((row["text"], word))
                    break
                    
    print(f"Found {len(mislabeled)} potentially mislabeled rows:")
    for text, word in mislabeled[:20]:
        print(f"- [Word: {word}] Text: {text}")
else:
    print("train.csv not found")
