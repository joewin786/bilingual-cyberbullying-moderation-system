import json
import logging
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_xlmr import CyberbullyDataset, generate_eval_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

def run_evaluation():
    model_dir = PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model"
    results_dir = PROJECT_ROOT / "outputs"
    
    if not model_dir.exists():
        logger.error(f"Model tidak ditemukan di {model_dir}")
        return

    # Load threshold dari training_results.json
    results_path = results_dir / "training_results.json"
    best_threshold = 0.50
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            res_data = json.load(f)
        best_threshold = res_data.get("threshold_tuning", {}).get("best_threshold", 0.50)
        logger.info(f"Loaded tuned threshold: {best_threshold}")
    
    logger.info("Memuat model dan tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model dimuat di device: {device}")
    
    # Setup dummy Trainer untuk evaluasi
    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "temp"),
        per_device_eval_batch_size=32,
        fp16=(device == "cuda"),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    for split_name, file_name in [("Validation", "val.csv"), ("Test", "test.csv")]:
        csv_path = PROJECT_ROOT / "data" / "processed" / file_name
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        logger.info(f"\nMengevaluasi {split_name} set ({len(df)} sampel)...")
        
        dataset = CyberbullyDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=128,
        )
        
        # Evaluasi menggunakan threshold optimal
        report = generate_eval_report(
            trainer, dataset,
            df["label"].tolist(), split_name, results_dir,
            threshold=best_threshold,
        )
        
        # Simpan metrik terbaru ke training_results.json agar sinkron
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            
            metric_key = f"{split_name.lower()}_metrics"
            res_data[metric_key] = report["metrics"]
            
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(res_data, f, indent=2, ensure_ascii=False)
                
    logger.info("\nEvaluasi cepat selesai! Laporan evaluasi terbaru telah disimpan.")

if __name__ == "__main__":
    run_evaluation()
