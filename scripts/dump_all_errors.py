import os
import sys
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import CyberbullyingPredictor

def dump_errors():
    model_dir = PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model"
    predictor = CyberbullyingPredictor(model_dir=model_dir)
    
    for file_name in ["test.csv", "val.csv"]:
        csv_path = PROJECT_ROOT / "data" / "processed" / file_name
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        labels_true = df["label"].astype(int).tolist()
        
        predictions = predictor.predict_batch(texts)
        
        errors = []
        for idx, (true_lbl, pred_res) in enumerate(zip(labels_true, predictions)):
            pred_lbl = pred_res.label_id
            if true_lbl != pred_lbl:
                errors.append({
                    "index": idx,
                    "text": texts[idx],
                    "true_label": true_lbl,
                    "pred_label": pred_lbl,
                    "confidence_bully": round(pred_res.confidence_bully, 4)
                })
                
        output_path = PROJECT_ROOT / "data" / f"all_errors_{file_name.replace('.csv', '.json')}"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"Dumped {len(errors)} errors for {file_name} to {output_path}")

if __name__ == "__main__":
    dump_errors()
