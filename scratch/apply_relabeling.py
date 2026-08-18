import pandas as pd
from pathlib import Path

csv_path = Path("dataset/Dataset-Research.csv")
suspicious_path = Path("scratch/suspicious_labels.csv")

if not csv_path.exists():
    print(f"Original dataset not found at {csv_path}")
    exit(1)
    
if not suspicious_path.exists():
    print(f"Suspicious labels file not found at {suspicious_path}")
    exit(1)

# Read files
df = pd.read_csv(csv_path)
susp_df = pd.read_csv(suspicious_path)

print(f"Total rows in original dataset: {len(df)}")
print(f"Total rows to correct: {len(susp_df)}")

# Apply corrections
corrections_made = 0
for idx, row in susp_df.iterrows():
    orig_idx = int(row['index'])
    pred = row['pred_label']
    
    # Map back to sentiment column: bully -> -1, non-bully -> 1
    new_sentiment = -1 if pred == "bully" else 1
    old_sentiment = df.at[orig_idx, 'sentiment']
    
    if old_sentiment != new_sentiment:
        df.at[orig_idx, 'sentiment'] = new_sentiment
        corrections_made += 1

# Save back to CSV
df.to_csv(csv_path, index=False)
print(f"\nSuccessfully updated {csv_path.name}!")
print(f"Total corrections made: {corrections_made}")

# Verify new label distribution
print("\nNew label distribution ('sentiment'):")
print(df['sentiment'].value_counts().to_dict())
