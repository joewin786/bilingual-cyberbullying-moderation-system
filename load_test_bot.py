"""
load_test_bot.py
=================
Skrip pengujian beban (load testing) untuk sistem deteksi cyberbullying
real-time (FastAPI + n8n + Discord bot).

Skrip ini mengirim sejumlah pesan uji ke endpoint API klasifikasi,
lalu mengukur tiga dimensi performa yang sudah didefinisikan di laporan:
  1. Latency (end-to-end, P50/P95/P99, mean, std dev)
  2. Throughput (pesan diproses per menit)
  3. Reliability (success rate, error count)

CARA PAKAI:
    pip install httpx
    python load_test_bot.py --url http://localhost:8000/predict --n 300 --mode burst

MODE PENGUJIAN:
  - sequential : kirim satu per satu, jeda otomatis menunggu respons
                 selesai sebelum kirim berikutnya. Cocok untuk mengukur
                 latency dalam kondisi wajar/tidak terbebani.
  - burst      : kirim banyak pesan bersamaan (concurrency diatur lewat
                 --concurrency). Cocok untuk menguji throughput dan
                 ketahanan sistem di bawah tekanan.

SIMULASI GANGGUAN (fault tolerance):
  Gunakan --timeout dengan nilai sangat kecil (misal 0.05) untuk
  mensimulasikan kondisi jaringan lambat/API tidak responsif, lalu amati
  apakah mekanisme fallback pada sistem bot (di luar skrip ini) benar-benar
  aktif ketika request gagal/timeout.

PENTING — SESUAIKAN DENGAN API ASLI:
  Fungsi build_payload() dan parse_response() di bawah ini memakai asumsi
  skema JSON generik. Sesuaikan nama field-nya dengan skema request/response
  FastAPI yang sebenarnya dipakai pada sistem Joewin.
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
# 1. KONFIGURASI DEFAULT — SESUAIKAN DENGAN API ASLI
# ------------------------------------------------------------------
DEFAULT_ENDPOINT = "http://localhost:8000/predict"  # ganti sesuai endpoint FastAPI
DEFAULT_TIMEOUT = 10.0  # detik


def build_payload(text: str) -> dict:
    """Sesuaikan struktur payload ini dengan skema request API asli."""
    return {"text": text}


def parse_response(data: dict) -> dict:
    """Sesuaikan key ini dengan struktur JSON respons API asli."""
    return {
        "label": data.get("label", data.get("prediction")),
        "probability": data.get("probability", data.get("confidence")),
    }


# ------------------------------------------------------------------
# 2. POOL PESAN UJI — variasi bahasa & tingkat toksisitas
#    Silakan tambah/ganti contoh agar merepresentasikan pola pesan
#    yang benar-benar muncul di server Discord asli.
# ------------------------------------------------------------------
SAMPLE_MESSAGES = {
    "id_aman": [
        "halo semua, semangat terus ya buat push rank malam ini",
        "makasih banget bantuannya tadi, sangat membantu",
        "ada yang mau mabar valorant jam 8 nanti?",
        "gg tim tadi solid banget mainnya",
    ],
    "id_toxic": [
        "dasar bocah bego main gini aja gabisa",
        "anjir noob banget sih lu, mending afk aja",
        "tolol emang lu, ngerusak game orang",
        "goblok banget cara main lu, keluar aja dari tim",
    ],
    "en_aman": [
        "gg everyone, that was a fun match",
        "thanks for carrying the team today",
        "anyone up for a scrim later tonight?",
        "nice play on that last round!",
    ],
    "en_toxic": [
        "you are so dumb, uninstall the game already",
        "worst player i have ever seen, quit already",
        "stop feeding you idiot, you're ruining it for everyone",
        "get lost, nobody wants you on this team",
    ],
    "mix_aman": [
        "gg tadi, good game banget deh seru",
        "thanks ya udah carry, appreciate it banget",
        "let's mabar again besok malam ya guys",
    ],
    "mix_toxic": [
        "goblok banget lu, so bad at this game",
        "dasar noob, you are useless banget dah",
        "stop main kalo emang gabisa, so annoying",
    ],
}


def sample_pool(n: int) -> list[str]:
    pool = [msg for msgs in SAMPLE_MESSAGES.values() for msg in msgs]
    return [random.choice(pool) for _ in range(n)]


@dataclass
class RequestResult:
    index: int
    text: str
    latency_ms: float
    status_code: int
    success: bool
    label: str = None
    error: str = None
    timestamp: str = None


async def send_one(client: httpx.AsyncClient, url: str, timeout: float,
                    index: int, text: str, results: list):
    payload = build_payload(text)
    start = time.perf_counter()
    ts = datetime.now().isoformat()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000
        success = resp.status_code == 200
        label = None
        if success:
            try:
                label = parse_response(resp.json()).get("label")
            except Exception:
                pass
        results.append(RequestResult(index, text, latency_ms, resp.status_code,
                                      success, label, None, ts))
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(RequestResult(index, text, latency_ms, 0, False, None,
                                      str(e), ts))


async def run_sequential(url: str, timeout: float, messages: list[str]) -> list:
    results = []
    async with httpx.AsyncClient() as client:
        for i, text in enumerate(messages):
            await send_one(client, url, timeout, i, text, results)
    return results


async def run_burst(url: str, timeout: float, messages: list[str],
                     concurrency: int) -> list:
    results = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def bound_send(i, text):
            async with sem:
                await send_one(client, url, timeout, i, text, results)

        await asyncio.gather(*[bound_send(i, t) for i, t in enumerate(messages)])
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


def build_report(results: list[RequestResult], duration_seconds: float) -> dict:
    latencies = [r.latency_ms for r in results if r.success]
    total = len(results)
    success_count = sum(r.success for r in results)
    error_count = total - success_count

    return {
        "total_pesan": total,
        "berhasil": success_count,
        "gagal": error_count,
        "success_rate_pct": round(100 * success_count / total, 2) if total else 0,
        "latency_p50_ms": round(percentile(latencies, 50), 2) if latencies else None,
        "latency_p95_ms": round(percentile(latencies, 95), 2) if latencies else None,
        "latency_p99_ms": round(percentile(latencies, 99), 2) if latencies else None,
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "latency_stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else None,
        "throughput_pesan_per_menit": round(total / (duration_seconds / 60), 2)
            if duration_seconds > 0 else 0,
        "durasi_uji_detik": round(duration_seconds, 2),
    }


def save_results(results: list[RequestResult], report: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out / f"load_test_raw_{ts_str}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    json_path = out / f"load_test_report_{ts_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def print_report(report: dict, mode: str):
    print("=" * 60)
    print(f"HASIL PENGUJIAN BEBAN — MODE: {mode.upper()}")
    print("=" * 60)
    print(f"Total pesan dikirim   : {report['total_pesan']}")
    print(f"Berhasil / Gagal      : {report['berhasil']} / {report['gagal']}")
    print(f"Success rate          : {report['success_rate_pct']}%")
    print(f"Durasi pengujian      : {report['durasi_uji_detik']} detik")
    print("-" * 60)
    print("LATENSI END-TO-END")
    print(f"  P50 (Median)        : {report['latency_p50_ms']} ms")
    print(f"  P95                 : {report['latency_p95_ms']} ms")
    print(f"  P99                 : {report['latency_p99_ms']} ms")
    print(f"  Rata-rata           : {report['latency_mean_ms']} ms")
    print(f"  Std Dev             : {report['latency_stdev_ms']} ms")
    print("-" * 60)
    print(f"THROUGHPUT            : {report['throughput_pesan_per_menit']} pesan/menit")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Load test untuk sistem deteksi cyberbullying real-time"
    )
    parser.add_argument("--url", default=DEFAULT_ENDPOINT,
                         help="Endpoint API klasifikasi (default: %(default)s)")
    parser.add_argument("--n", type=int, default=300,
                         help="Jumlah pesan uji (default: 300)")
    parser.add_argument("--mode", choices=["sequential", "burst"], default="burst",
                         help="sequential = satu-satu, burst = bersamaan (default: burst)")
    parser.add_argument("--concurrency", type=int, default=20,
                         help="Jumlah request bersamaan pada mode burst (default: 20)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                         help="Timeout per request dalam detik. Perkecil nilai ini "
                              "untuk mensimulasikan gangguan jaringan (default: 10.0)")
    parser.add_argument("--output", default="./outputs",
                         help="Folder penyimpanan hasil (default: ./outputs)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed agar pemilihan pesan uji dapat direproduksi")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    messages = sample_pool(args.n)

    print(f"Mengirim {args.n} pesan ke {args.url} (mode: {args.mode}, "
          f"timeout: {args.timeout}s)...")

    start = time.perf_counter()
    if args.mode == "sequential":
        results = asyncio.run(run_sequential(args.url, args.timeout, messages))
    else:
        results = asyncio.run(run_burst(args.url, args.timeout, messages, args.concurrency))
    duration = time.perf_counter() - start

    report = build_report(results, duration)
    print_report(report, args.mode)

    csv_path, json_path = save_results(results, report, args.output)
    print(f"\nHasil mentah (per-request)  : {csv_path}")
    print(f"Ringkasan report (JSON)      : {json_path}")


if __name__ == "__main__":
    main()