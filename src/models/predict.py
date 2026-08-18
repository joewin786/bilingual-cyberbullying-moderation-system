"""
predict.py
==========
Utility modul untuk menjalankan prediksi tunggal atau batch
menggunakan model XLM-RoBERTa yang sudah di-fine-tune.

Digunakan oleh:
- api/main.py (inference server)
- Script testing manual
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Add project root to sys.path if not present to ensure relative imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.text_normalizer import normalize_text

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "xlmr_cyberbully" / "best_model"

# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    text: str
    label: str                  # "bully" | "non-bully"
    label_id: int               # 1 | 0
    confidence: float           # confidence for predicted label (0.0–1.0)
    confidence_bully: float     # raw score untuk kelas bully
    confidence_non_bully: float # raw score untuk kelas non-bully
    action_tier: str            # "ignore" | "flag" | "action"
    severity: str = "non_bullying"  # "non_bullying" | "mild" | "moderate" | "severe"

    def to_dict(self) -> dict:
        return {
            "text":                 self.text,
            "label":                self.label,
            "label_id":             self.label_id,
            "confidence":           round(self.confidence, 4),
            "confidence_bully":     round(self.confidence_bully, 4),
            "confidence_non_bully": round(self.confidence_non_bully, 4),
            "action_tier":          self.action_tier,
            "severity":             self.severity,
        }


# ──────────────────────────────────────────────────────────────
# Predictor Class
# ──────────────────────────────────────────────────────────────
class CyberbullyingPredictor:
    """
    Wrapper untuk model XLM-RoBERTa fine-tuned.
    Load sekali, prediksi berkali-kali (thread-safe untuk single process).
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        max_length: int = 128,
        confidence_flag: float = 0.70,      # >= flag → log ke moderator
        confidence_action: float = 0.85,    # >= action → aksi otomatis
        # Severity thresholds — confidence_bully mapping:
        #   < threshold_mild                 → non_bullying
        #   threshold_mild  ≤ conf < moderate → mild
        #   threshold_moderate ≤ conf < severe → moderate
        #   ≥ threshold_severe               → severe
        threshold_mild: float = 0.70,
        threshold_moderate: float = 0.80,
        threshold_severe: float = 0.92,
        device: str = None,
        classification_threshold: float = None,  # Custom classification threshold
    ):
        self.model_dir         = Path(model_dir)
        self.max_length        = max_length
        self.confidence_flag   = confidence_flag
        self.confidence_action = confidence_action
        self.threshold_mild     = threshold_mild
        self.threshold_moderate = threshold_moderate
        self.threshold_severe   = threshold_severe

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load dynamic classification threshold from training results
        if classification_threshold is not None:
            self.classification_threshold = classification_threshold
        else:
            self.classification_threshold = 0.50  # Default fallback
            try:
                results_path = PROJECT_ROOT / "outputs" / "training_results.json"
                if results_path.exists():
                    import json
                    with open(results_path, "r", encoding="utf-8") as f:
                        res_data = json.load(f)
                    best_thr = res_data.get("threshold_tuning", {}).get("best_threshold", 0.50)
                    self.classification_threshold = float(best_thr)
                    logger.info(f"Loaded classification threshold: {self.classification_threshold} from training_results.json")
                else:
                    logger.info(f"training_results.json tidak ditemukan, menggunakan threshold: {self.classification_threshold}")
            except Exception as e:
                logger.warning(f"Gagal memuat threshold: {e}. Menggunakan default: {self.classification_threshold}")

        self._load_model()

    def _load_model(self):
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {self.model_dir}. "
                "Jalankan src/models/train_xlmr.py terlebih dahulu."
            )

        logger.info(f"Memuat model dari {self.model_dir} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        # Fine Tuned Transformer based classifier
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.id2label = self.model.config.id2label   # {0: "non-bully", 1: "bully"}
        logger.info(f"Model berhasil dimuat. Device: {self.device}")

    def _get_action_tier(self, label: str, confidence: float) -> str:
        """Tentukan tier tindakan berdasarkan label dan confidence."""
        if label == "non-bully":
            return "ignore"
        # label == "bully"
        if confidence < self.confidence_flag:
            return "ignore"
        elif confidence < self.confidence_action:
            return "flag"
        else:
            return "action"

    def _get_severity(self, label: str, confidence_bully: float) -> str:
        """
        Petakan confidence_bully ke severity level multi-kelas.

        Mapping (berdasarkan threshold configurable):
          non_bullying  → label non-bully ATAU confidence_bully < threshold_mild
          mild          → threshold_mild ≤ conf < threshold_moderate
          moderate      → threshold_moderate ≤ conf < threshold_severe
          severe        → conf ≥ threshold_severe

        Args:
            label: "bully" atau "non-bully"
            confidence_bully: probabilitas kelas bully (0.0–1.0)

        Returns:
            str severity level
        """
        if label == "non-bully" or confidence_bully < self.threshold_mild:
            return "non_bullying"
        elif confidence_bully < self.threshold_moderate:
            return "mild"
        elif confidence_bully < self.threshold_severe:
            return "moderate"
        else:
            return "severe"

    def predict(self, text: str) -> PredictionResult:
        """Prediksi satu teks."""
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str]) -> List[PredictionResult]:
        """Prediksi batch teks (lebih efisien untuk multiple input)."""
        if not texts:
            return []

        # Apply Indonesian slang and spelling normalization before tokenization
        normalized_texts = [normalize_text(t) for t in texts]

        # Tokenisasi
        inputs = self.tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # Inferensi
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()  # shape: (N, 2)

        results = []
        for i, text in enumerate(texts):
            prob_non_bully = float(probs[i][0])
            prob_bully     = float(probs[i][1])

            # Apply custom classification threshold
            label_id   = 1 if prob_bully >= self.classification_threshold else 0
            label      = self.id2label[label_id]
            confidence = prob_bully if label_id == 1 else prob_non_bully

            tier     = self._get_action_tier(label, prob_bully)
            severity = self._get_severity(label, prob_bully)

            results.append(PredictionResult(
                text=text, # Keep original text in results
                label=label,
                label_id=label_id,
                confidence=confidence,
                confidence_bully=prob_bully,
                confidence_non_bully=prob_non_bully,
                action_tier=tier,
                severity=severity,
            ))

        return results


# ──────────────────────────────────────────────────────────────
# Quick Test (CLI)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test prediksi model")
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--text",      type=str, default=None)
    args = parser.parse_args()

    predictor = CyberbullyingPredictor(model_dir=args.model_dir)

    test_sentences = [
        args.text if args.text else "kamu bodoh sekali tidak ada gunanya",
        "semangat ya! kamu pasti bisa",
        "dasar anjing bangsat mati saja kamu",
        "cyberbullying adalah perilaku yang menyakiti orang lain secara online",   # edukasi
        "you are so stupid and worthless, go die",
        "have a great day everyone!",
    ]

    print("\n" + "=" * 65)
    print("CYBERBULLYING DETECTION — TEST PREDIKSI")
    print("=" * 65)
    for sentence in test_sentences:
        result = predictor.predict(sentence)
        print(f"\n  Teks       : {sentence[:60]}...")
        print(f"  Label      : {result.label.upper()} (id={result.label_id})")
        print(f"  Confidence : bully={result.confidence_bully:.3f} | non-bully={result.confidence_non_bully:.3f}")
        print(f"  Tier       : {result.action_tier.upper()}")
    print("=" * 65)
