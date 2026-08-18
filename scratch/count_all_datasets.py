import os
import pandas as pd
from pathlib import Path

dataset_dir = Path("dataset")

def count_rows(file_path):
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'utf-8-sig', 'cp1252']:
        try:
            # First check separator
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                header = f.readline()
            sep = ';' if ';' in header and ',' not in header else ','
            
            df = pd.read_csv(file_path, encoding=encoding, sep=sep, on_bad_lines='skip')
            return len(df), list(df.columns)
        except Exception as e:
            continue
    return None, None

results = []
print("Searching for CSV datasets under 'dataset/'...")
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.csv'):
            full_path = Path(root) / file
            # Skip lexicons or small helper tables if they are not datasets
            is_helper = "lexicon" in file.lower() or "stopwords" in file.lower() or "singkatan" in file.lower()
            
            row_count, cols = count_rows(full_path)
            if row_count is not None:
                results.append({
                    "path": full_path.as_posix(),
                    "name": file,
                    "rows": row_count,
                    "columns": cols,
                    "is_helper": is_helper
                })

# Print results
print("\n=== RAW DATASET COUNT ===")
total_rows = 0
total_helper_rows = 0
for r in results:
    type_str = "[Helper/Lexicon]" if r["is_helper"] else "[Dataset]"
    print(f"- {r['path']} ({type_str}): {r['rows']} rows | Columns: {r['columns']}")
    if r["is_helper"]:
        total_helper_rows += r["rows"]
    else:
        total_rows += r["rows"]

print("\n=========================")
print(f"Total Raw Dataset Rows: {total_rows}")
print(f"Total Helper/Lexicon Rows: {total_helper_rows}")
