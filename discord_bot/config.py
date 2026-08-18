"""
discord_bot/config.py
=====================
Konfigurasi Discord Bot dari environment variables (.env file)
dan moderation_config.yaml untuk action mapping per severity.
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env dari root project
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)


def _load_moderation_config(path: str) -> dict:
    """Load moderation_config.yaml. Return empty dict jika file tidak ada."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"moderation_config.yaml tidak ditemukan di {path}. Pakai default.")
        return {}
    except Exception as e:
        logger.error(f"Gagal load moderation_config.yaml: {e}")
        return {}


# Load moderation config sekali saat module di-import
_MOD_CONFIG_PATH = str(
    Path(__file__).resolve().parents[1] / "configs" / "moderation_config.yaml"
)
_MOD_CFG = _load_moderation_config(_MOD_CONFIG_PATH)


class Config:
    # ── Discord ──────────────────────────────────────────────
    DISCORD_TOKEN: str   = os.getenv("DISCORD_TOKEN", "")
    GUILD_ID: int        = int(os.getenv("GUILD_ID", "0"))
    MOD_CHANNEL_ID: int  = int(os.getenv("MOD_CHANNEL_ID", "0"))
    MUTE_ROLE_ID: int    = int(os.getenv("MUTE_ROLE_ID", "0"))

    # ── n8n Webhook ──────────────────────────────────────────
    N8N_WEBHOOK_URL: str = os.getenv(
        "N8N_WEBHOOK_URL",
        "http://localhost:5678/webhook/cyberbully"
    )

    # ── API ──────────────────────────────────────────────────
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")

    # ── Moderasi (legacy fallback) ────────────────────────────
    MUTE_DURATION_HOURS: int  = int(os.getenv("MUTE_DURATION_HOURS", "8"))
    VIOLATION_RESET_DAYS: int = int(os.getenv("VIOLATION_RESET_DAYS", "30"))

    # ── Thresholds ───────────────────────────────────────────
    CONFIDENCE_FLAG: float   = float(os.getenv("CONFIDENCE_FLAG", "0.70"))
    CONFIDENCE_ACTION: float = float(os.getenv("CONFIDENCE_ACTION", "0.85"))

    # ── Database ─────────────────────────────────────────────
    POSTGRES_HOST: str     = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int     = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str       = os.getenv("POSTGRES_DB", "cyberbully_db")
    POSTGRES_USER: str     = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")

    # ── Feature Flags ────────────────────────────────────────
    # Prefix pesan yang dikecualikan dari deteksi (misalnya: perintah bot)
    EXEMPT_PREFIXES: list = ["!", "/", "."]

    # Bot tidak memeriksa pesan dari bot lain
    IGNORE_BOTS: bool = True

    # ── Moderation Config (dari YAML) ─────────────────────────
    MODERATION_CONFIG_PATH: str = _MOD_CONFIG_PATH

    # Severity thresholds (dari YAML, fallback ke hardcode)
    @classmethod
    def get_severity_thresholds(cls) -> dict:
        """Ambil threshold severity dari moderation_config.yaml."""
        defaults = {"mild": 0.70, "moderate": 0.80, "severe": 0.92}
        return _MOD_CFG.get("severity_thresholds", defaults)

    # Action config per severity
    @classmethod
    def get_action_config(cls, severity: str) -> dict:
        """Ambil konfigurasi tindakan untuk severity tertentu."""
        actions = _MOD_CFG.get("actions", {})
        defaults = {
            "mild":     {"action": "warn",    "dm": True, "duration_minutes": 0},
            "moderate": {"action": "timeout", "dm": True, "duration_minutes": 30},
            "severe":   {"action": "timeout", "dm": True, "duration_minutes": 480,
                         "require_mod_approval": False},
        }
        return actions.get(severity, defaults.get(severity, {}))

    # Mod review config
    @classmethod
    def get_mod_review_config(cls) -> dict:
        """Ambil konfigurasi mod review."""
        defaults = {
            "low_confidence_threshold": 0.75,
            "always_notify_severity": ["severe"],
            "interactive_buttons": True,
        }
        return _MOD_CFG.get("mod_review", defaults)

    # Escalation config
    @classmethod
    def get_escalation_rules(cls) -> list:
        """Ambil aturan eskalasi berdasarkan jumlah pelanggaran."""
        esc = _MOD_CFG.get("escalation", {})
        if not esc.get("enabled", True):
            return []
        return esc.get("rules", [])

    @classmethod
    def validate(cls):
        """Validasi konfigurasi wajib."""
        errors = []
        if not cls.DISCORD_TOKEN:
            errors.append("DISCORD_TOKEN belum diset di .env")
        if cls.GUILD_ID == 0:
            errors.append("GUILD_ID belum diset di .env")
        if cls.MOD_CHANNEL_ID == 0:
            errors.append("MOD_CHANNEL_ID belum diset di .env")
        if cls.MUTE_ROLE_ID == 0:
            errors.append("MUTE_ROLE_ID belum diset di .env")
        if errors:
            raise EnvironmentError(
                "Konfigurasi tidak lengkap:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return True
