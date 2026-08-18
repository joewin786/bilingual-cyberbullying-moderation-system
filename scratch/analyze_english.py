import pandas as pd
from pathlib import Path

processed_dir = Path("data/processed")
en_train_path = processed_dir / "en_train.csv"
en_test_path = processed_dir / "en_test.csv"
en_val_path = processed_dir / "en_val.csv"

print("=== English Dataset Stats ===")
for name, p in [("Train", en_train_path), ("Val", en_val_path), ("Test", en_test_path)]:
    if p.exists():
        df = pd.read_csv(p)
        print(f"{name} set:")
        print(f"  Total samples: {len(df)}")
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
    else:
        print(f"{name} set does not exist at {p}")
