import pandas as pd
import re

df = pd.read_csv("data/processed/train.csv")

def get_matching_words(substring, texts):
    matching_words = set()
    pattern = re.compile(rf"\b\w*{substring}\w*\b", re.IGNORECASE)
    for text in texts:
        if not isinstance(text, str):
            continue
        for word in text.split():
            clean_word = re.sub(r"[^\w]", "", word)
            if pattern.search(clean_word):
                matching_words.add(clean_word.lower())
    return list(matching_words)[:20]

names = ["Budi", "Andi", "Ani", "Joe", "Micu"]

print("=== Unique Words Matching Substrings ===")
for name in names:
    matches = get_matching_words(name, df["text"])
    print(f"Name: {name} | Matching words: {matches}")
