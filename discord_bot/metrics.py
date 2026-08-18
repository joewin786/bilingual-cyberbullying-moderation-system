"""
discord_bot/metrics.py
======================
Modul untuk mengumpulkan dan menghitung metrik performa bot secara real-time.

Metrik yang dikumpulkan:
  - Latency      : end-to-end, API, dan n8n (P50, P95, P99)
  - Throughput   : pesan/menit, total pesan diproses
  - Reliability  : API success rate, n8n success rate, fallback rate, error count, uptime
"""

import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional


class MetricsTracker:
    """
    Thread-safe in-memory tracker untuk metrik performa bot.

    Menggunakan sliding window (deque) berukuran MAX_WINDOW_SIZE untuk
    menghitung persentil latency tanpa konsumsi memori tak terbatas.
    """

    MAX_WINDOW_SIZE = 1000   # Jumlah request terbaru yang disimpan

    def __init__(self):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        # ── Latency windows (menyimpan nilai ms) ──────────────────
        self._total_latencies: deque[float] = deque(maxlen=self.MAX_WINDOW_SIZE)
        self._api_latencies:   deque[float] = deque(maxlen=self.MAX_WINDOW_SIZE)
        self._n8n_latencies:   deque[float] = deque(maxlen=self.MAX_WINDOW_SIZE)

        # ── Throughput counters ────────────────────────────────────
        self._total_messages: int = 0          # semua pesan yang diproses
        self._total_detections: int = 0        # tier != "ignore"
        # Ring buffer waktu (unix timestamp) untuk hitung per-menit
        self._message_timestamps: deque[float] = deque(maxlen=500)
        self._detection_timestamps: deque[float] = deque(maxlen=500)

        # ── Reliability counters ───────────────────────────────────
        self._api_requests: int = 0
        self._api_successes: int = 0
        self._n8n_requests: int = 0
        self._n8n_successes: int = 0
        self._fallback_count: int = 0          # n8n gagal → pakai API langsung
        self._error_count: int = 0             # exception / unexpected error

    # ──────────────────────────────────────────────────────────────
    # Record Methods
    # ──────────────────────────────────────────────────────────────
    def record_message(
        self,
        total_latency_ms: float,
        tier: str,
        used_fallback: bool = False,
    ):
        """
        Catat setiap pesan yang berhasil diproses.

        Args:
            total_latency_ms: Waktu total dari menerima pesan hingga aksi selesai (ms)
            tier: "ignore" | "flag" | "action"
            used_fallback: True jika n8n gagal dan pakai API direct
        """
        now = time.monotonic()
        with self._lock:
            self._total_latencies.append(total_latency_ms)
            self._total_messages += 1
            self._message_timestamps.append(now)

            if tier != "ignore":
                self._total_detections += 1
                self._detection_timestamps.append(now)

            if used_fallback:
                self._fallback_count += 1

    def record_api_call(self, latency_ms: float, success: bool):
        """Catat hasil panggilan ke FastAPI /predict."""
        with self._lock:
            self._api_latencies.append(latency_ms)
            self._api_requests += 1
            if success:
                self._api_successes += 1

    def record_n8n_call(self, latency_ms: float, success: bool):
        """Catat hasil panggilan ke n8n webhook."""
        with self._lock:
            self._n8n_latencies.append(latency_ms)
            self._n8n_requests += 1
            if success:
                self._n8n_successes += 1

    def record_error(self):
        """Tambah 1 ke error counter."""
        with self._lock:
            self._error_count += 1

    # ──────────────────────────────────────────────────────────────
    # Compute Methods
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _percentile(data: list[float], p: int) -> Optional[float]:
        """Hitung persentil ke-p dari list data. Return None jika data kosong."""
        if not data:
            return None
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        idx = min(idx, len(sorted_data) - 1)
        return round(sorted_data[idx], 1)

    @staticmethod
    def _per_minute(timestamps: deque, window_seconds: int = 60) -> float:
        """Hitung rate (event/menit) dari timestamp ring buffer."""
        now = time.monotonic()
        cutoff = now - window_seconds
        count = sum(1 for t in timestamps if t >= cutoff)
        return round(count / (window_seconds / 60), 1)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format detik menjadi string yang mudah dibaca."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}j {m}m {s}d"
        elif m > 0:
            return f"{m}m {s}d"
        return f"{s}d"

    def snapshot(self) -> dict:
        """
        Ambil snapshot semua metrik saat ini.
        Return dict siap pakai untuk ditampilkan ke user.
        """
        with self._lock:
            total_lat = list(self._total_latencies)
            api_lat   = list(self._api_latencies)
            n8n_lat   = list(self._n8n_latencies)

            api_success_rate = (
                round(self._api_successes / self._api_requests * 100, 1)
                if self._api_requests > 0 else None
            )
            n8n_success_rate = (
                round(self._n8n_successes / self._n8n_requests * 100, 1)
                if self._n8n_requests > 0 else None
            )
            fallback_rate = (
                round(self._fallback_count / self._n8n_requests * 100, 1)
                if self._n8n_requests > 0 else 0.0
            )

            uptime_s = time.monotonic() - self._start_time

            return {
                # Latency (ms)
                "total_p50": self._percentile(total_lat, 50),
                "total_p95": self._percentile(total_lat, 95),
                "total_p99": self._percentile(total_lat, 99),
                "total_avg": round(sum(total_lat) / len(total_lat), 1) if total_lat else None,
                "api_p50":   self._percentile(api_lat, 50),
                "api_p95":   self._percentile(api_lat, 95),
                "n8n_p50":   self._percentile(n8n_lat, 50),
                "n8n_p95":   self._percentile(n8n_lat, 95),

                # Throughput
                "total_messages":    self._total_messages,
                "total_detections":  self._total_detections,
                "msg_per_minute":    self._per_minute(self._message_timestamps),
                "det_per_minute":    self._per_minute(self._detection_timestamps),

                # Reliability
                "api_requests":      self._api_requests,
                "api_successes":     self._api_successes,
                "api_success_rate":  api_success_rate,
                "n8n_requests":      self._n8n_requests,
                "n8n_successes":     self._n8n_successes,
                "n8n_success_rate":  n8n_success_rate,
                "fallback_count":    self._fallback_count,
                "fallback_rate":     fallback_rate,
                "error_count":       self._error_count,

                # System
                "uptime_seconds":    uptime_s,
                "uptime_str":        self._format_duration(uptime_s),
                "window_size":       len(total_lat),
                "max_window_size":   self.MAX_WINDOW_SIZE,
                "snapshot_at":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }


# Singleton — diimpor oleh bot.py
metrics = MetricsTracker()
