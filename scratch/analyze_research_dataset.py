import pandas as pd
from pathlib import Path

csv_path = Path("dataset/Dataset-Research.csv")

try:
    df = pd.read_csv(csv_path)
    print("=== Dataset-Research.csv Info ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution ('sentiment'):")
    print(df['sentiment'].value_counts().to_dict())
    
    print("\nMissing values:")
    print(df.isna().sum().to_dict())
    
    print("\nDuplicates based on comment:")
    print(df.duplicated(subset=['comment']).sum())
    
    print("\n--- Example positive/neutral (1) ---")
    pos_samples = df[df['sentiment'] == 1].head(10)
    for idx, row in pos_samples.iterrows():
         print(f"- {row['comment']}")
         
    print("\n--- Example negative (-1) ---")
    neg_samples = df[df['sentiment'] == -1].head(10)
    for idx, row in neg_samples.iterrows():
         print(f"- {row['comment']}")
         
except Exception as e:
    print(f"Error reading CSV: {e}")
