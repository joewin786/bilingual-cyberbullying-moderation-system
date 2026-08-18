"""
discord_bot/database.py
=======================
PostgreSQL database untuk menyimpan riwayat pelanggaran user dan log tindakan bot.

Tabel:
    violations   — jumlah pelanggaran per user per server
    action_logs  — log semua tindakan bot (untuk audit)
    appeals      — log permintaan banding (appeal) dari user
    performance_logs  — histori performa sistem
    mod_review_queue  — antrian review manual moderator
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor

from .config import Config

logger = logging.getLogger(__name__)


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    """
    Wrapper PostgreSQL database untuk bot moderasi.
    Semua operasi synchronous (dijalankan di thread pool oleh discord.py).
    """

    def __init__(self):
        self._init_db()

    @contextmanager
    def _connect(self):
        """
        Koneksi context manager untuk PostgreSQL.
        Membuka koneksi baru, mengembalikan cursor, meng-commit di akhir,
        melakukan rollback jika error, dan menutup koneksi secara bersih.
        """
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise e
        finally:
            cur.close()
            conn.close()

    def _init_db(self):
        """Buat tabel jika belum ada."""
        with self._connect() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id                  SERIAL PRIMARY KEY,
                    user_id             VARCHAR(100) NOT NULL,
                    guild_id            VARCHAR(100) NOT NULL,
                    violation_count     INTEGER DEFAULT 0,
                    last_violation_at   VARCHAR(100),
                    muted_until         VARCHAR(100),
                    UNIQUE(user_id, guild_id)
                );

                CREATE TABLE IF NOT EXISTS action_logs (
                    id              SERIAL PRIMARY KEY,
                    user_id         VARCHAR(100) NOT NULL,
                    guild_id        VARCHAR(100) NOT NULL,
                    channel_id      VARCHAR(100),
                    message_id      VARCHAR(100),
                    action          VARCHAR(100) NOT NULL,
                    message_content TEXT,
                    confidence      REAL,
                    action_tier     VARCHAR(100),
                    severity        VARCHAR(100),
                    violation_count INTEGER,
                    timestamp       VARCHAR(100) NOT NULL
                );

                CREATE TABLE IF NOT EXISTS appeals (
                    id              SERIAL PRIMARY KEY,
                    user_id         VARCHAR(100) NOT NULL,
                    guild_id        VARCHAR(100) NOT NULL,
                    message_id      VARCHAR(100),
                    reason          TEXT,
                    status          VARCHAR(100) DEFAULT 'pending',
                    reviewed_by     VARCHAR(100),
                    review_note     TEXT,
                    timestamp       VARCHAR(100) NOT NULL,
                    reviewed_at     VARCHAR(100)
                );

                CREATE TABLE IF NOT EXISTS performance_logs (
                    id                  SERIAL PRIMARY KEY,
                    guild_id            VARCHAR(100),
                    timestamp           VARCHAR(100) NOT NULL,
                    total_messages      INTEGER,
                    total_detections    INTEGER,
                    avg_latency_ms      REAL,
                    p95_latency_ms      REAL,
                    p99_latency_ms      REAL,
                    api_success_rate    REAL,
                    n8n_success_rate    REAL,
                    fallback_rate       REAL,
                    error_count         INTEGER,
                    uptime_seconds      REAL
                );

                CREATE TABLE IF NOT EXISTS mod_review_queue (
                    id              SERIAL PRIMARY KEY,
                    guild_id        VARCHAR(100) NOT NULL,
                    user_id         VARCHAR(100) NOT NULL,
                    channel_id      VARCHAR(100),
                    message_id      VARCHAR(100),
                    message_content TEXT,
                    confidence      REAL,
                    severity        VARCHAR(100),
                    action_tier     VARCHAR(100),
                    status          VARCHAR(100) DEFAULT 'pending',
                    reviewed_by     VARCHAR(100),
                    review_note     TEXT,
                    timestamp       VARCHAR(100) NOT NULL,
                    reviewed_at     VARCHAR(100),
                    embed_message_id VARCHAR(100)
                );

                CREATE INDEX IF NOT EXISTS idx_violations_user
                    ON violations(user_id, guild_id);

                CREATE INDEX IF NOT EXISTS idx_logs_user
                    ON action_logs(user_id, guild_id);

                CREATE INDEX IF NOT EXISTS idx_perf_logs_ts
                    ON performance_logs(timestamp);

                CREATE INDEX IF NOT EXISTS idx_mod_review_status
                    ON mod_review_queue(status, guild_id);

                CREATE INDEX IF NOT EXISTS idx_appeals_status
                    ON appeals(status, guild_id);
            """)

            # Migrations
            cur.execute("ALTER TABLE action_logs ADD COLUMN IF NOT EXISTS severity VARCHAR(100)")
            cur.execute("ALTER TABLE appeals ADD COLUMN IF NOT EXISTS review_note TEXT")
            cur.execute("ALTER TABLE appeals ADD COLUMN IF NOT EXISTS reviewed_at VARCHAR(100)")

        logger.info(f"Database diinisialisasi pada PostgreSQL: {Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}")

    # ──────────────────────────────────────────────────────
    # Violations
    # ──────────────────────────────────────────────────────
    def get_violations(self, user_id: int, guild_id: int) -> Optional[dict]:
        """Ambil data pelanggaran user. Return None jika belum ada."""
        with self._connect() as cur:
            cur.execute(
                "SELECT * FROM violations WHERE user_id=%s AND guild_id=%s",
                (str(user_id), str(guild_id)),
            )
            row = cur.fetchone()
        return row

    def increment_violation(self, user_id: int, guild_id: int) -> int:
        """
        Tambah 1 pelanggaran. Reset jika sudah lebih dari VIOLATION_RESET_DAYS.
        Return: jumlah pelanggaran saat ini.
        """
        now = get_utc_now().isoformat()
        reset_threshold = (
            get_utc_now() - timedelta(days=Config.VIOLATION_RESET_DAYS)
        ).isoformat()

        with self._connect() as cur:
            cur.execute(
                "SELECT * FROM violations WHERE user_id=%s AND guild_id=%s",
                (str(user_id), str(guild_id)),
            )
            existing = cur.fetchone()

            if existing is None:
                # Insert baru
                cur.execute(
                    """
                    INSERT INTO violations (user_id, guild_id, violation_count, last_violation_at)
                    VALUES (%s, %s, 1, %s)
                    """,
                    (str(user_id), str(guild_id), now),
                )
                count = 1
            else:
                last_violation = existing["last_violation_at"]

                # Reset jika sudah lama
                if last_violation and last_violation < reset_threshold:
                    new_count = 1
                    logger.info(
                        f"Violation count direset untuk user {user_id} "
                        f"(terakhir: {last_violation})"
                    )
                else:
                    new_count = existing["violation_count"] + 1

                cur.execute(
                    """
                    UPDATE violations
                    SET violation_count=%s, last_violation_at=%s
                    WHERE user_id=%s AND guild_id=%s
                    """,
                    (new_count, now, str(user_id), str(guild_id)),
                )
                count = new_count

        return count

    def set_muted_until(self, user_id: int, guild_id: int, muted_until: datetime):
        """Set waktu mute berakhir untuk user."""
        with self._connect() as cur:
            cur.execute(
                """
                UPDATE violations
                SET muted_until=%s
                WHERE user_id=%s AND guild_id=%s
                """,
                (muted_until.isoformat(), str(user_id), str(guild_id)),
            )

    def is_muted(self, user_id: int, guild_id: int) -> Tuple[bool, Optional[datetime]]:
        """
        Cek apakah user masih dalam masa mute.
        Return: (is_muted, muted_until)
        """
        row = self.get_violations(user_id, guild_id)
        if row is None or row["muted_until"] is None:
            return False, None

        muted_until = datetime.fromisoformat(row["muted_until"])
        if muted_until.tzinfo is None:
            muted_until = muted_until.replace(tzinfo=timezone.utc)

        if get_utc_now() < muted_until:
            return True, muted_until
        return False, None

    def reset_violations(self, user_id: int, guild_id: int):
        """Reset semua pelanggaran user (digunakan oleh admin)."""
        with self._connect() as cur:
            cur.execute(
                """
                UPDATE violations
                SET violation_count=0, muted_until=NULL
                WHERE user_id=%s AND guild_id=%s
                """,
                (str(user_id), str(guild_id)),
            )
        logger.info(f"Violations direset untuk user {user_id} di guild {guild_id}")

    # ──────────────────────────────────────────────────────
    # Action Logs
    # ──────────────────────────────────────────────────────
    def log_action(
        self,
        user_id: int,
        guild_id: int,
        action: str,
        message_content: str = None,
        confidence: float = None,
        action_tier: str = None,
        severity: str = None,
        violation_count: int = None,
        channel_id: int = None,
        message_id: int = None,
    ):
        """Simpan log tindakan bot."""
        with self._connect() as cur:
            cur.execute(
                """
                INSERT INTO action_logs
                    (user_id, guild_id, channel_id, message_id, action,
                     message_content, confidence, action_tier, severity,
                     violation_count, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(user_id), str(guild_id),
                    str(channel_id) if channel_id else None,
                    str(message_id) if message_id else None,
                    action, message_content, confidence, action_tier, severity,
                    violation_count, get_utc_now().isoformat(),
                ),
            )

    def get_user_logs(self, user_id: int, guild_id: int, limit: int = 10) -> list:
        """Ambil log tindakan terbaru untuk user tertentu."""
        with self._connect() as cur:
            cur.execute(
                """
                SELECT * FROM action_logs
                WHERE user_id=%s AND guild_id=%s
                ORDER BY timestamp DESC LIMIT %s
                """,
                (str(user_id), str(guild_id), limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_guild_stats(self, guild_id: int) -> dict:
        """Statistik deteksi untuk server."""
        with self._connect() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM action_logs WHERE guild_id=%s",
                (str(guild_id),),
            )
            total = list(cur.fetchone().values())[0]

            cur.execute(
                """
                SELECT action, COUNT(*) as cnt
                FROM action_logs WHERE guild_id=%s
                GROUP BY action
                """,
                (str(guild_id),),
            )
            actions = cur.fetchall()

            cur.execute(
                "SELECT COUNT(DISTINCT user_id) FROM action_logs WHERE guild_id=%s",
                (str(guild_id),),
            )
            unique_users = list(cur.fetchone().values())[0]

        stats = {
            "total_detections": total,
            "unique_users_flagged": unique_users,
            "actions": {r["action"]: r["cnt"] for r in actions},
        }
        return stats

    # ──────────────────────────────────────────────────────
    # Performance Logs
    # ──────────────────────────────────────────────────────
    def save_performance_snapshot(self, guild_id: int, snapshot: dict):
        """Simpan snapshot metrik performa ke database untuk histori."""
        with self._connect() as cur:
            cur.execute(
                """
                INSERT INTO performance_logs
                    (guild_id, timestamp, total_messages, total_detections,
                     avg_latency_ms, p95_latency_ms, p99_latency_ms,
                     api_success_rate, n8n_success_rate, fallback_rate,
                     error_count, uptime_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(guild_id) if guild_id else None,
                    snapshot.get("snapshot_at"),
                    snapshot.get("total_messages"),
                    snapshot.get("total_detections"),
                    snapshot.get("total_avg"),
                    snapshot.get("total_p95"),
                    snapshot.get("total_p99"),
                    snapshot.get("api_success_rate"),
                    snapshot.get("n8n_success_rate"),
                    snapshot.get("fallback_rate"),
                    snapshot.get("error_count"),
                    snapshot.get("uptime_seconds"),
                ),
            )
        logger.debug(f"Performance snapshot disimpan untuk guild {guild_id}")

    def get_performance_history(self, guild_id: int = None, limit: int = 24) -> list:
        """Ambil histori snapshot performa (default: 24 entri terakhir = 24 jam)."""
        with self._connect() as cur:
            if guild_id:
                cur.execute(
                    """
                    SELECT * FROM performance_logs
                    WHERE guild_id=%s
                    ORDER BY timestamp DESC LIMIT %s
                    """,
                    (str(guild_id), limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM performance_logs
                    ORDER BY timestamp DESC LIMIT %s
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ──────────────────────────────────────────────────────
    # Appeals
    # ──────────────────────────────────────────────────────
    def add_appeal(
        self,
        user_id: int,
        guild_id: int,
        message_id: int = None,
        reason: str = None,
    ):
        with self._connect() as cur:
            cur.execute(
                """
                INSERT INTO appeals (user_id, guild_id, message_id, reason, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    str(user_id), str(guild_id),
                    str(message_id) if message_id else None,
                    reason, get_utc_now().isoformat(),
                ),
            )
        logger.info(f"Appeal diterima dari user {user_id}")

    def get_pending_appeals(self, guild_id: int, limit: int = 20) -> list:
        """Ambil semua appeal yang masih pending untuk server ini."""
        with self._connect() as cur:
            cur.execute(
                """
                SELECT * FROM appeals
                WHERE guild_id=%s AND status='pending'
                ORDER BY timestamp DESC LIMIT %s
                """,
                (str(guild_id), limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def resolve_appeal(
        self,
        appeal_id: int,
        reviewed_by: int,
        status: str,
        note: str = None,
    ):
        """
        Selesaikan appeal (approve atau reject).
        status: 'approved' | 'rejected'
        """
        with self._connect() as cur:
            cur.execute(
                """
                UPDATE appeals
                SET status=%s, reviewed_by=%s, review_note=%s, reviewed_at=%s
                WHERE id=%s
                """,
                (status, str(reviewed_by), note, get_utc_now().isoformat(), appeal_id),
            )
        logger.info(f"Appeal {appeal_id} diselesaikan: {status} oleh {reviewed_by}")

    # ──────────────────────────────────────────────────────
    # Mod Review Queue
    # ──────────────────────────────────────────────────────
    def add_to_review_queue(
        self,
        guild_id: int,
        user_id: int,
        message_content: str,
        confidence: float,
        severity: str,
        action_tier: str,
        channel_id: int = None,
        message_id: int = None,
        embed_message_id: int = None,
    ) -> int:
        """
        Tambah entry ke antrian mod review.
        Return: ID entry yang baru dibuat.
        """
        with self._connect() as cur:
            cur.execute(
                """
                INSERT INTO mod_review_queue
                    (guild_id, user_id, channel_id, message_id, message_content,
                     confidence, severity, action_tier, timestamp, embed_message_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    str(guild_id), str(user_id),
                    str(channel_id) if channel_id else None,
                    str(message_id) if message_id else None,
                    message_content, confidence, severity, action_tier,
                    get_utc_now().isoformat(),
                    str(embed_message_id) if embed_message_id else None,
                ),
            )
            inserted_id = cur.fetchone()["id"]
        return inserted_id

    def get_pending_reviews(self, guild_id: int, limit: int = 20) -> list:
        """Ambil antrian review yang masih pending."""
        with self._connect() as cur:
            cur.execute(
                """
                SELECT * FROM mod_review_queue
                WHERE guild_id=%s AND status='pending'
                ORDER BY timestamp DESC LIMIT %s
                """,
                (str(guild_id), limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def resolve_review(
        self,
        review_id: int,
        reviewed_by: int,
        status: str,
        note: str = None,
    ):
        """
        Selesaikan item dari mod_review_queue.
        status: 'approved' (lanjut eksekusi) | 'rejected' (batalkan) | 'dismissed' (abaikan)
        """
        with self._connect() as cur:
            cur.execute(
                """
                UPDATE mod_review_queue
                SET status=%s, reviewed_by=%s, review_note=%s, reviewed_at=%s
                WHERE id=%s
                """,
                (status, str(reviewed_by), note, get_utc_now().isoformat(), review_id),
            )
        logger.info(f"Review {review_id} diselesaikan: {status} oleh {reviewed_by}")

    def update_review_embed_id(self, review_id: int, embed_message_id: int):
        """Update embed_message_id di mod_review_queue (setelah embed dikirim ke channel)."""
        with self._connect() as cur:
            cur.execute(
                "UPDATE mod_review_queue SET embed_message_id=%s WHERE id=%s",
                (str(embed_message_id), review_id),
            )

    def get_review_by_id(self, review_id: int) -> Optional[dict]:
        """Ambil data detail mod_review_queue berdasarkan ID."""
        with self._connect() as cur:
            cur.execute("SELECT * FROM mod_review_queue WHERE id=%s", (review_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def get_appeal_by_id(self, appeal_id: int) -> Optional[dict]:
        """Ambil data detail appeal berdasarkan ID."""
        with self._connect() as cur:
            cur.execute("SELECT * FROM appeals WHERE id=%s", (appeal_id,))
            row = cur.fetchone()
        return dict(row) if row else None

