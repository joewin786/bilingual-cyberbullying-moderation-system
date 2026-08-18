import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_xlmr import CyberbullyDataset

def test_thresholds():
    model_dir = PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model"
    csv_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    
    if not model_dir.exists() or not csv_path.exists():
        print("Model atau data uji tidak ditemukan!")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    df = pd.read_csv(csv_path)
    dataset = CyberbullyDataset(
        texts=df["text"].tolist(),
        labels=df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=128,
    )
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="models/temp", per_device_eval_batch_size=32, fp16=(device=="cuda")),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels_true = df["label"].tolist()
    
    probs = softmax(logits, axis=-1)
    bully_probs = probs[:, 1]
    
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    print("\n[INFO] PERBANDINGAN TINGKAT FP & FN PADA BERBAGAI THRESHOLD (TEST SET)")
    print("=" * 75)
    print(f" {'Threshold':^9} | {'Akurasi':^8} | {'F1 Macro':^9} | {'FP (Salah Tuduh)':^16} | {'FN (Lolos)':^12}")
    print("-" * 75)
    
    for thr in thresholds:
        preds = (bully_probs >= thr).astype(int)
        acc = accuracy_score(labels_true, preds)
        _, _, f1_mac, _ = precision_recall_fscore_support(labels_true, preds, average="macro", zero_division=0)
        
        cm = confusion_matrix(labels_true, preds)
        fp = cm[0][1]  # True:0, Pred:1
        fn = cm[1][0]  # True:1, Pred:0
        
        marker = " <- Saat ini" if thr == 0.65 else ""
        print(f"   {thr:>7.2f} |  {acc:>6.4f} |   {f1_mac:>6.4f}  |      {fp:>5}       |    {fn:>5}    {marker}")
    print("=" * 75)

if __name__ == "__main__":
    test_thresholds()
