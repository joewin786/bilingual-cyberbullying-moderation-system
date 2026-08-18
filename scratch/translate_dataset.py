import pandas as pd
import torch
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Paths
en_train_path = Path("data/processed/en_train.csv")
out_path = Path("dataset/translated_en_to_id.csv")

if not en_train_path.exists():
    print(f"Error: {en_train_path} not found!")
    exit(1)

print("Loading English dataset...")
df_en = pd.read_csv(en_train_path)
print(f"Total English training samples: {len(df_en)}")

# Sample 5,000 samples (2,500 bully [1], 2,500 non-bully [0])
df_bully = df_en[df_en['label'] == 1]
df_non_bully = df_en[df_en['label'] == 0]

print(f"Available English bully samples: {len(df_bully)}")
print(f"Available English non-bully samples: {len(df_non_bully)}")

sample_size = min(2500, len(df_bully), len(df_non_bully))
print(f"Sampling {sample_size * 2} samples ({sample_size} bully, {sample_size} non-bully)...")

df_sampled = pd.concat([
    df_bully.sample(n=sample_size, random_state=42),
    df_non_bully.sample(n=sample_size, random_state=42)
]).sample(frac=1.0, random_state=42).reset_index(drop=True) # Shuffle

# Load translation model
model_name = "Helsinki-NLP/opus-mt-en-id"
print(f"Loading translation model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

# Clear CUDA cache
if device == "cuda":
    torch.cuda.empty_cache()

texts = df_sampled['text'].astype(str).tolist()
labels = df_sampled['label'].tolist()

translated_texts = []
batch_size = 32 # Reduced batch size for safety

print("Starting translation (optimized with greedy search)...")
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i : i + batch_size]
    
    # Preprocess batch: clean possible empty texts and limit length to avoid huge computations
    batch_texts = [t[:400] if t.strip() else "empty" for t in batch_texts]
    
    # Tokenize
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate translation (Optimized parameters)
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=1,       # Greedy search (very fast!)
            early_stopping=True
        )
        
    # Decode
    batch_translated = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    translated_texts.extend(batch_translated)

# Save results
df_translated = pd.DataFrame({
    'text': translated_texts,
    'label': labels
})

# Save to CSV
df_translated.to_csv(out_path, index=False, encoding='utf-8')
print(f"\nSuccessfully translated dataset and saved to {out_path}!")
print(f"Total rows: {len(df_translated)}")
print(f"Label distribution:")
print(df_translated['label'].value_counts().to_dict())
