"""
api/main.py
===========
FastAPI Inference Server untuk Cyberbullying Detection.

Endpoints:
    GET  /health                → cek status server dan model
    POST /predict               → prediksi satu teks
    POST /predict/batch         → prediksi batch teks
    GET  /metrics/snapshot      → snapshot evaluasi bot terbaru (untuk n8n)
    GET  /metrics/history       → riwayat snapshot evaluasi bot (untuk n8n)

Cara jalankan:
    cd api
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Atau dari root project:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH         = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / cfg["training"]["output_dir"])))
CONFIDENCE_FLAG    = float(os.getenv("CONFIDENCE_FLAG",   cfg["inference"]["confidence_ignore"]))
CONFIDENCE_ACTION  = float(os.getenv("CONFIDENCE_ACTION", cfg["inference"]["confidence_flag"]))

# Severity thresholds (bisa di-override via env var)
THRESHOLD_MILD     = float(os.getenv("THRESHOLD_MILD",     "0.70"))
THRESHOLD_MODERATE = float(os.getenv("THRESHOLD_MODERATE", "0.80"))
THRESHOLD_SEVERE   = float(os.getenv("THRESHOLD_SEVERE",   "0.92"))

MAX_BATCH_SIZE     = 32

# PostgreSQL Database config
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB       = os.getenv("POSTGRES_DB", "cyberbully_db")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

# ──────────────────────────────────────────────────────────────
# Global Predictor (loaded once at startup)
# ──────────────────────────────────────────────────────────────
predictor = None
model_load_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat startup, cleanup saat shutdown."""
    global predictor, model_load_time

    logger.info("=" * 50)
    logger.info("Cyberbullying Detection API — Starting up")
    logger.info(f"Model path     : {MODEL_PATH}")
    logger.info(f"Conf. flag     : {CONFIDENCE_FLAG}")
    logger.info(f"Conf. action   : {CONFIDENCE_ACTION}")
    logger.info("=" * 50)

    try:
        from src.models.predict import CyberbullyingPredictor
        t0 = time.time()
        predictor = CyberbullyingPredictor(
            model_dir=MODEL_PATH,
            max_length=cfg["model"]["max_length"],
            confidence_flag=CONFIDENCE_FLAG,
            confidence_action=CONFIDENCE_ACTION,
            threshold_mild=THRESHOLD_MILD,
            threshold_moderate=THRESHOLD_MODERATE,
            threshold_severe=THRESHOLD_SEVERE,
        )
        model_load_time = round(time.time() - t0, 2)
        logger.info(f"✅ Model berhasil dimuat dalam {model_load_time}s")
        logger.info(
            f"Severity thresholds: mild≥{THRESHOLD_MILD} | "
            f"moderate≥{THRESHOLD_MODERATE} | severe≥{THRESHOLD_SEVERE}"
        )
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.warning("Server berjalan TANPA model (mode degraded). Endpoint /predict tidak tersedia.")

    yield  # Server berjalan di sini

    logger.info("Server shutting down...")


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cyberbullying Detection API",
    description=(
        "Deteksi cyberbullying bilingual (Indonesia + Inggris) "
        "menggunakan XLM-RoBERTa fine-tuned."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048, description="Teks yang ingin diprediksi")
    lang: Optional[str] = Field("id", description="Bahasa: 'id' (Indonesia) atau 'en' (Inggris)")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Teks tidak boleh kosong atau hanya spasi")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    lang: Optional[str] = Field("id")


class PredictionResponse(BaseModel):
    text: str
    label: str                      # "bully" | "non-bully"
    label_id: int                   # 1 | 0
    confidence: float               # confidence prediksi label
    confidence_bully: float         # probabilitas bully
    confidence_non_bully: float     # probabilitas non-bully
    action_tier: str                # "ignore" | "flag" | "action"
    severity: str                   # "non_bullying" | "mild" | "moderate" | "severe"
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_load_time_s: Optional[float]
    confidence_flag: float
    confidence_action: float
    threshold_mild: float
    threshold_moderate: float
    threshold_severe: float
    version: str


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────
def ensure_model_loaded():
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum dimuat. Pastikan fine-tuning sudah selesai dan "
                f"model tersedia di {MODEL_PATH}. "
                "Jalankan: python src/models/train_xlmr.py"
            ),
        )


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Cyberbullying Detection API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Cek status server dan apakah model sudah dimuat."""
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        model_path=str(MODEL_PATH),
        model_load_time_s=model_load_time,
        confidence_flag=CONFIDENCE_FLAG,
        confidence_action=CONFIDENCE_ACTION,
        threshold_mild=THRESHOLD_MILD,
        threshold_moderate=THRESHOLD_MODERATE,
        threshold_severe=THRESHOLD_SEVERE,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Prediksi satu teks apakah mengandung cyberbullying.

    **Action Tier:**
    - `ignore`  → confidence < 70% (bukan bully atau tidak yakin)
    - `flag`    → 70–85% (kirim ke moderator untuk review manual)
    - `action`  → > 85% (aksi otomatis: warn/mute/kick)
    """
    ensure_model_loaded()

    t0 = time.perf_counter()
    result = predictor.predict(request.text)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(
        f"PREDICT | label={result.label} | tier={result.action_tier} "
        f"| conf_bully={result.confidence_bully:.3f} | {elapsed_ms}ms "
        f"| text={request.text[:50]}..."
    )

    return PredictionResponse(
        text=result.text,
        label=result.label,
        label_id=result.label_id,
        confidence=result.confidence,
        confidence_bully=result.confidence_bully,
        confidence_non_bully=result.confidence_non_bully,
        action_tier=result.action_tier,
        severity=result.severity,
        processing_time_ms=elapsed_ms,
    )


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Prediksi batch teks (maksimal 32 teks sekaligus).
    Lebih efisien untuk multiple request.
    """
    ensure_model_loaded()

    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maksimal {MAX_BATCH_SIZE} teks per request."
        )

    t0 = time.perf_counter()
    results = predictor.predict_batch(request.texts)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    per_item   = round(elapsed_ms / len(results), 2)

    logger.info(
        f"BATCH PREDICT | n={len(results)} | total={elapsed_ms}ms | per_item={per_item}ms"
    )

    return [
        PredictionResponse(
            text=r.text,
            label=r.label,
            label_id=r.label_id,
            confidence=r.confidence,
            confidence_bully=r.confidence_bully,
            confidence_non_bully=r.confidence_non_bully,
            action_tier=r.action_tier,
            severity=r.severity,
            processing_time_ms=per_item,
        )
        for r in results
    ]


# ──────────────────────────────────────────────────────────────
# Metrics Endpoints (untuk n8n)
# ──────────────────────────────────────────────────────────────
def _read_performance_logs(limit: int = 1) -> List[Dict[str, Any]]:
    """
    Baca performance_logs dari PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM performance_logs ORDER BY timestamp DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"Gagal membaca performance_logs dari PostgreSQL: {e}")
        return []


@app.get("/metrics/snapshot", tags=["Metrics"])
async def get_metrics_snapshot():
    """
    Ambil snapshot performa bot paling terbaru dari database.

    Digunakan oleh **n8n** untuk polling data evaluasi sistem secara terjadwal.
    Data bersumber dari tabel `performance_logs` yang ditulis oleh Discord bot
    setiap 30 menit (auto-save) atau setiap kali command `/metrics` dipanggil.

    **Fields yang dikembalikan:**
    - `snapshot`: Row terbaru (latency P95/P99, throughput, reliability, uptime)
    - `data_age_minutes`: Selisih waktu sekarang vs timestamp snapshot (freshness indicator)
    - `is_stale`: True jika data lebih dari 60 menit yang lalu
    - `retrieved_at`: Waktu query ini dieksekusi (UTC)
    """
    rows = _read_performance_logs(limit=1)

    if not rows:
        return {
            "snapshot": None,
            "data_age_minutes": None,
            "is_stale": True,
            "message": "Belum ada data performa. Pastikan Discord bot sudah berjalan.",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    latest = rows[0]

    # Hitung freshness data
    try:
        ts_str = latest.get("timestamp", "")
        # Handle format dengan atau tanpa timezone
        if ts_str.endswith("UTC"):
            ts_str = ts_str.replace(" UTC", "+00:00")
        snapshot_time = datetime.fromisoformat(ts_str)
        if snapshot_time.tzinfo is None:
            snapshot_time = snapshot_time.replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        age_minutes = round((now_utc - snapshot_time).total_seconds() / 60, 1)
        is_stale = age_minutes > 60
    except (ValueError, TypeError):
        age_minutes = None
        is_stale = True

    return {
        "snapshot": latest,
        "data_age_minutes": age_minutes,
        "is_stale": is_stale,
        "message": "ok" if not is_stale else "Data mungkin sudah usang (> 60 menit).",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics/history", tags=["Metrics"])
async def get_metrics_history(
    limit: int = Query(default=24, ge=1, le=168, description="Jumlah entri (max 168 = 1 minggu jika interval 1 jam)"),
):
    """
    Ambil riwayat snapshot performa bot (N entri terbaru).

    Berguna untuk melihat tren latency/throughput dari waktu ke waktu.
    Default: 24 entri terakhir (≈ 24 jam jika interval 1 jam, atau ≈ 12 jam jika 30 menit).
    """
    rows = _read_performance_logs(limit=limit)

    if not rows:
        return {
            "history": [],
            "count": 0,
            "message": "Belum ada data performa. Pastikan Discord bot sudah berjalan.",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }

    # Hitung agregat sederhana dari histori
    valid_latencies = [r["p95_latency_ms"] for r in rows if r.get("p95_latency_ms") is not None]
    valid_api_rates = [r["api_success_rate"] for r in rows if r.get("api_success_rate") is not None]

    summary = {
        "avg_p95_latency_ms": round(sum(valid_latencies) / len(valid_latencies), 1) if valid_latencies else None,
        "max_p95_latency_ms": max(valid_latencies) if valid_latencies else None,
        "avg_api_success_rate": round(sum(valid_api_rates) / len(valid_api_rates), 1) if valid_api_rates else None,
        "total_messages_latest": rows[0].get("total_messages") if rows else None,
        "total_detections_latest": rows[0].get("total_detections") if rows else None,
    }

    return {
        "history": rows,
        "count": len(rows),
        "summary": summary,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
