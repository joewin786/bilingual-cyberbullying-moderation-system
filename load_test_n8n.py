"""
load_test_n8n.py
================
Skrip pengujian beban (load testing) langsung ke Webhook Production/Test n8n
(Sistem Deteksi Cyberbullying).

Skrip ini mengirim hingga 300 pesan (atau di bawah 300 pesan sesuai argumen --n)
langsung ke endpoint webhook n8n, lalu menghitung latensi, throughput, dan success rate.

CARA PAKAI:
  1. Jalankan mode Production (Default 300 pesan, burst mode):
     python load_test_n8n.py

  2. Jalankan dengan jumlah pesan spesifik (misal 100 pesan):
     python load_test_n8n.py --n 100

  3. Jalankan mode Sequential (satu per satu):
     python load_test_n8n.py --n 50 --mode sequential

  4. Jalankan ke Webhook Test n8n (saat tombol 'Test Step' di UI n8n aktif):
     python load_test_n8n.py --n 50 --test-mode
"""

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import httpx

# ------------------------------------------------------------------
# KONFIGURASI ENDPOINT N8N WEBHOOK
# ------------------------------------------------------------------
DEFAULT_PROD_URL = "http://localhost:5678/webhook/cyberbully"
DEFAULT_TEST_URL = "http://localhost:5678/webhook-test/cyberbully"
DEFAULT_TIMEOUT = 15.0  # detik (sesuai timeout node n8n)

# ------------------------------------------------------------------
# POOL PESAN UJI (Bilingual Indonesia & Inggris, Toxic & Safe)
# ------------------------------------------------------------------
SAMPLE_MESSAGES = [
    {"text": "halo semua, semangat terus ya buat push rank malam ini", "lang": "id"},
    {"text": "makasih banget bantuannya tadi, sangat membantu", "lang": "id"},
    {"text": "ada yang mau mabar valorant jam 8 nanti?", "lang": "id"},
    {"text": "gg tim tadi solid banget mainnya", "lang": "id"},
    {"text": "dasar bocah bego main gini aja gabisa", "lang": "id"},
    {"text": "anjir noob banget sih lu, mending afk aja", "lang": "id"},
    {"text": "tolol emang lu, ngerusak game orang", "lang": "id"},
    {"text": "goblok banget cara main lu, keluar aja dari tim", "lang": "id"},
    {"text": "gg everyone, that was a fun match", "lang": "en"},
    {"text": "thanks for carrying the team today", "lang": "en"},
    {"text": "anyone up for a scrim later tonight?", "lang": "en"},
    {"text": "nice play on that last round!", "lang": "en"},
    {"text": "you are so dumb, uninstall the game already", "lang": "en"},
    {"text": "worst player i have ever seen, quit already", "lang": "en"},
    {"text": "stop feeding you idiot, you're ruining it for everyone", "lang": "en"},
    {"text": "get lost, nobody wants you on this team", "lang": "en"},
    {"text": "gg tadi, good game banget deh seru", "lang": "id"},
    {"text": "goblok banget lu, so bad at this game", "lang": "id"},
]


def sample_pool(n: int) -> list[dict]:
    """Mengambil n sampel acak dari dataset pesan uji."""
    return [random.choice(SAMPLE_MESSAGES) for _ in range(n)]


@dataclass
class RequestResult:
    index: int
    text: str
    lang: str
    latency_ms: float
    status_code: int
    success: bool
    response_data: str = None
    error: str = None
    timestamp: str = None


async def send_one(client: httpx.AsyncClient, url: str, timeout: float,
                    index: int, item: dict, results: list):
    payload = {"text": item["text"], "lang": item.get("lang", "id")}
    start = time.perf_counter()
    ts = datetime.now().isoformat()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        success = resp.status_code in (200, 201)
        resp_text = resp.text[:200] if resp.text else ""
        results.append(RequestResult(
            index=index,
            text=payload["text"],
            lang=payload["lang"],
            latency_ms=latency_ms,
            status_code=resp.status_code,
            success=success,
            response_data=resp_text,
            error=None if success else f"HTTP {resp.status_code}: {resp_text}",
            timestamp=ts
        ))
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(RequestResult(
            index=index,
            text=payload["text"],
            lang=payload["lang"],
            latency_ms=latency_ms,
            status_code=0,
            success=False,
            response_data="",
            error=str(e),
            timestamp=ts
        ))


async def run_sequential(url: str, timeout: float, messages: list[dict]) -> list:
    results = []
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(messages):
            await send_one(client, url, timeout, i, item, results)
            # sedikit jeda halus agar log mudah dibaca
            await asyncio.sleep(0.01)
    return results


async def run_burst(url: str, timeout: float, messages: list[dict], concurrency: int) -> list:
    results = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def bound_send(i, item):
            async with sem:
                await send_one(client, url, timeout, i, item, results)

        await asyncio.gather(*[bound_send(i, item) for i, item in enumerate(messages)])
    return results


def percentile(data: list[float], pct: float):
    if not data:
        return None
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def build_report(results: list[RequestResult], duration_seconds: float, target_url: str) -> dict:
    latencies = [r.latency_ms for r in results if r.success]
    total = len(results)
    success_count = sum(r.success for r in results)
    error_count = total - success_count

    return {
        "target_url": target_url,
        "total_pesan": total,
        "berhasil": success_count,
        "gagal": error_count,
        "success_rate_pct": round(100 * success_count / total, 2) if total else 0,
        "latency_p50_ms": round(percentile(latencies, 50), 2) if latencies else None,
        "latency_p95_ms": round(percentile(latencies, 95), 2) if latencies else None,
        "latency_p99_ms": round(percentile(latencies, 99), 2) if latencies else None,
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else None,
        "throughput_pesan_per_menit": round(total / (duration_seconds / 60), 2) if duration_seconds > 0 else 0,
        "durasi_uji_detik": round(duration_seconds, 2),
    }


def save_results(results: list[RequestResult], report: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out / f"n8n_load_test_raw_{ts_str}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    json_path = out / f"n8n_load_test_report_{ts_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def print_report(report: dict, mode: str):
    print("=" * 65)
    print(f"HASIL PENGUJIAN TEMBAK WEBHOOK N8N — MODE: {mode.upper()}")
    print("=" * 65)
    print(f"Target URL            : {report['target_url']}")
    print(f"Total pesan dikirim   : {report['total_pesan']}")
    print(f"Berhasil / Gagal      : {report['berhasil']} / {report['gagal']}")
    print(f"Success rate          : {report['success_rate_pct']}%")
    print(f"Durasi pengujian      : {report['durasi_uji_detik']} detik")
    print("-" * 65)
    print("LATENSI END-TO-END N8N WEBHOOK")
    print(f"  P50 (Median)        : {report['latency_p50_ms']} ms")
    print(f"  P95                 : {report['latency_p95_ms']} ms")
    print(f"  P99                 : {report['latency_p99_ms']} ms")
    print(f"  Rata-rata           : {report['latency_mean_ms']} ms")
    print(f"  Std Dev             : {report['latency_stdev_ms']} ms")
    print("-" * 65)
    print(f"THROUGHPUT            : {report['throughput_pesan_per_menit']} pesan/menit")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Skrip tembak pesan langsung ke n8n Production Webhook"
    )
    parser.add_argument("--url", default=None,
                        help="Custom URL webhook n8n (opsional)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Gunakan URL webhook-test alih-alih production")
    parser.add_argument("--n", type=int, default=300,
                        help="Jumlah pesan yang dikirim (default: 300, maks direkomendasikan <= 300)")
    parser.add_argument("--mode", choices=["burst", "sequential"], default="burst",
                        help="burst = kirim bersamaan dengan concurrency, sequential = kirim satu per satu (default: burst)")
    parser.add_argument("--concurrency", type=int, default=15,
                        help="Jumlah request paralel pada mode burst (default: 15)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="Timeout per request dalam detik (default: 15.0)")
    parser.add_argument("--output", default="./outputs",
                        help="Folder penyimpanan hasil laporan (default: ./outputs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed untuk konsistensi pesan (default: 42)")
    args = parser.parse_args()

    # Tentukan Target URL
    if args.url:
        target_url = args.url
    elif args.test_mode:
        target_url = DEFAULT_TEST_URL
    else:
        target_url = DEFAULT_PROD_URL

    if args.seed is not None:
        random.seed(args.seed)

    # Pastikan n <= 300 jika tidak diset secara khusus
    num_messages = max(1, args.n)
    messages = sample_pool(num_messages)

    print(f"\n[*] Memulai pengiriman {num_messages} pesan ke Webhook n8n...")
    print(f"[>] Target URL : {target_url}")
    print(f"[>] Mode       : {args.mode.upper()} (Concurrency: {args.concurrency} worker)" if args.mode == "burst" else f"[>] Mode       : SEQUENTIAL")
    print(f"[>] Timeout    : {args.timeout} detik\n")

    start = time.perf_counter()
    if args.mode == "sequential":
        results = asyncio.run(run_sequential(target_url, args.timeout, messages))
    else:
        results = asyncio.run(run_burst(target_url, args.timeout, messages, args.concurrency))
    duration = time.perf_counter() - start

    report = build_report(results, duration, target_url)
    print_report(report, args.mode)

    csv_path, json_path = save_results(results, report, args.output)
    print(f"\nData mentah per request tersimpan di : {csv_path}")
    print(f"Ringkasan laporan JSON tersimpan di  : {json_path}\n")


if __name__ == "__main__":
    main()
