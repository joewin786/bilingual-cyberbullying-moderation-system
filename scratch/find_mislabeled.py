import pandas as pd
import torch
from pathlib import Path
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = Path("models/xlmr_cyberbully/best_model")
csv_path = Path("dataset/Dataset-Research.csv")

if not model_dir.exists():
    print("Model directory not found!")
    exit(1)

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path.name}")

# Standardize labels for inspection: 1 = bully, 0 = non-bully
# In Dataset-Research.csv, sentiment -1 = bully, 1 = non-bully
df['original_label_binary'] = df['sentiment'].apply(lambda x: 1 if x == -1 else (0 if x == 1 else -1))

suspicious_samples = []

print("Running model predictions...")
batch_size = 32
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    texts = batch['comment'].tolist()
    
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu().numpy()
        probs = softmax(logits, axis=-1)
        
    for j, (idx, row) in enumerate(batch.iterrows()):
        pred_label_id = probs[j].argmax()
        confidence = probs[j][pred_label_id]
        orig_label = row['original_label_binary']
        
        # If prediction contradicts label and confidence is high
        if pred_label_id != orig_label and confidence > 0.85:
            suspicious_samples.append({
                "index": idx,
                "comment": row['comment'],
                "original_sentiment": row['sentiment'],
                "original_label": "bully" if orig_label == 1 else "non-bully",
                "pred_label": "bully" if pred_label_id == 1 else "non-bully",
                "confidence": confidence
            })

susp_df = pd.DataFrame(suspicious_samples)
print(f"\nFound {len(susp_df)} suspicious samples (pred != orig with confidence > 85%)")

if len(susp_df) > 0:
    susp_df.to_csv("scratch/suspicious_labels.csv", index=False)
    print("Saved suspicious samples to scratch/suspicious_labels.csv")
    
    print("\nSample suspicious comments (first 10):")
    for idx, row in susp_df.head(10).iterrows():
        print(f"Text: '{row['comment']}'")
        print(f"  Original Label: {row['original_label']} | Model Prediction: {row['pred_label']} ({row['confidence']:.2%})\n")
