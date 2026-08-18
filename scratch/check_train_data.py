import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"

if train_path.exists():
    df = pd.read_csv(train_path)
    print(f"Total train rows: {len(df)}")
    
    # Search for anjing
    anjing_df = df[df["text"].str.contains("anjing", case=False, na=False)]
    print(f"\nRows containing 'anjing': {len(anjing_df)}")
    if len(anjing_df) > 0:
        print(anjing_df["label"].value_counts())
        print(anjing_df.head(5))
        
    # Search for bangsat
    bangsat_df = df[df["text"].str.contains("bangsat", case=False, na=False)]
    print(f"\nRows containing 'bangsat': {len(bangsat_df)}")
    if len(bangsat_df) > 0:
        print(bangsat_df["label"].value_counts())
        print(bangsat_df.head(5))
else:
    print("train.csv not found")
