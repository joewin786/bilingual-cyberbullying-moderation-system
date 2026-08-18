import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.text_normalizer import normalize_text

text = "dasar anjing bangsat mati saja kamu"
normalized = normalize_text(text)
print(f"Original  : '{text}'")
print(f"Normalized: '{normalized}'")
