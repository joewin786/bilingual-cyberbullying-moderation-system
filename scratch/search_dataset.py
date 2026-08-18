import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
datasets = [
    "dataset/combined_dataset.csv",
    "dataset/dataset_clean.csv",
    "dataset/cyberbullying_dataset_1000.csv",
]

def search_terms():
    for ds_path in datasets:
        path = PROJECT_ROOT / ds_path
        if not path.exists():
            print(f"Skipping {ds_path} (does not exist)")
            continue
            
        print(f"\n=== Searching in {ds_path} ===")
        df = pd.read_csv(path)
        print(f"Columns: {df.columns.tolist()}")
        
        # Try to find the text column
        text_col = None
        for col in df.columns:
            if "text" in col.lower() or "string" in col.lower() or "tweet" in col.lower():
                text_col = col
                break
        if not text_col:
            text_col = df.columns[0]
            
        print(f"Using text column: '{text_col}'")
        
        # Search for 'argen'
        match_argen = df[df[text_col].astype(str).str.contains("argen", case=False)]
        if not match_argen.empty:
            print(f"Found {len(match_argen)} matches for 'argen':")
            label_col = "Label" if "Label" in df.columns else ("label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else None))
            print(match_argen[[text_col, label_col]].head(10))
        else:
            print("No matches for 'argen'")
            
        # Search for 'dataset'
        match_dataset = df[df[text_col].astype(str).str.contains("dataset", case=False)]
        if not match_dataset.empty:
            print(f"Found {len(match_dataset)} matches for 'dataset':")
            label_col = "Label" if "Label" in df.columns else ("label" if "label" in df.columns else (df.columns[1] if len(df.columns) > 1 else None))
            print(match_dataset[[text_col, label_col]].head(10))
        else:
            print("No matches for 'dataset'")

if __name__ == "__main__":
    search_terms()
