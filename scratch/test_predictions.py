import sys
import re
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from discord_bot.preprocessor import preprocess
from discord_bot.bot import is_likely_safe, contains_badword
from src.models.predict import CyberbullyingPredictor

def test_pipeline(text):
    print(f"\n--- Testing: '{text}' ---")
    
    # 1. Preprocess
    prep = preprocess(text)
    print(f"Original Text : {prep.original}")
    print(f"Cleaned Text  : {prep.text}")
    
    # 2. Check is_likely_safe
    safe = is_likely_safe(prep.text)
    print(f"Is Likely Safe (FP Guard): {safe}")
    
    # 3. Predict using model
    predictor = CyberbullyingPredictor()
    result = predictor.predict(prep.text)
    print(f"Model Label   : {result.label} (Confidence Bully: {result.confidence_bully:.4f})")
    
    # 4. Final Bot classification status
    will_skip = safe
    print(f"Will Bot Skip ?: {'YES (Safe)' if will_skip else 'NO (Processes for Moderation)'}")

if __name__ == "__main__":
    test_pipeline("nd ada di dataset wak, otomatis nd na detect ")
    test_pipeline("Hadeh pasti pendukung argen")
