"""
system_quality_tester.py
========================
Skrip pengujian komprehensif untuk 5 aspek kualitas sistem (NFR):
1. Scalability (Concurrency Sweep: 5, 20, 50, 100)
2. Reliability & Consistency (Uji determinisme prediksi 20x)
3. Fault Tolerance & Recovery (Uji respons saat service down & ukur MTTR)
4. Availability (Polling healthcheck & hitung uptime %)

PENGGUNAAN:
  python system_quality_tester.py --all
  python system_quality_tester.py --test scalability
  python system_quality_tester.py --test consistency
  python system_quality_tester.py --test fault
  python system_quality_tester.py --test availability
"""

import argparse
import asyncio
import csv
import json
import random
import statistics
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import httpx

# Konfigurasi Endpoint Default
API_PREDICT_URL = "http://localhost:8000/predict"
API_HEALTH_URL = "http://localhost:8000/health"
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/cyberbully"

OUTPUT_DIR = Path("./outputs/quality_evaluation")

# Sampel Pesan Uji
SAMPLE_MESSAGES = [
    {"text": "halo semua, semangat terus ya buat push rank malam ini", "lang": "id"},
    {"text": "dasar bocah bego main gini aja gabisa", "lang": "id"},
    {"text": "anjir noob banget sih lu, mending afk aja", "lang": "id"},
    {"text": "gg everyone, that was a fun match", "lang": "en"},
    {"text": "you are so dumb, uninstall the game already", "lang": "en"},
    {"text": "worst player i have ever seen, quit already", "lang": "en"},
    {"text": "makasih banget bantuannya tadi, sangat membantu", "lang": "id"},
    {"text": "goblok banget lu, so bad at this game", "lang": "id"},
]

def percentile(data: list[float], pct: float):
    if not data:
        return 0.0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


# =====================================================================
# 1. SCALABILITY SWEEP TEST
# =====================================================================
async def _send_single(client: httpx.AsyncClient, url: str, payload: dict, timeout: float, method: str = "POST"):
    start = time.perf_counter()
    try:
        if method.upper() == "GET":
            resp = await client.get(url, timeout=timeout)
        else:
            resp = await client.post(url, json=payload, timeout=timeout)
        lat = (time.perf_counter() - start) * 1000
        return resp.status_code == 200, lat, resp.status_code, resp.text
    except Exception as e:
        lat = (time.perf_counter() - start) * 1000
        return False, lat, 0, str(e)


async def run_scalability_sweep(url: str = API_PREDICT_URL, n_per_level: int = 200):
    print("=" * 65)
    print(" 1. SCALABILITY CONCURRENCY SWEEP TEST")
    print("=" * 65)
    
    concurrency_levels = [5, 20, 50, 100]
    sweep_results = {}
    
    for conc in concurrency_levels:
        print(f"[*] Menguji Concurrency Level = {conc} (Total {n_per_level} pesan)...")
        sem = asyncio.Semaphore(conc)
        pool = [random.choice(SAMPLE_MESSAGES) for _ in range(n_per_level)]
        latencies = []
        successes = 0
        failures = 0
        
        async with httpx.AsyncClient() as client:
            async def worker(item):
                nonlocal successes, failures
                async with sem:
                    ok, lat, code, _ = await _send_single(client, url, {"text": item["text"]}, 15.0)
                    latencies.append(lat)
                    if ok:
                        successes += 1
                    else:
                        failures += 1

            start_t = time.perf_counter()
            await asyncio.gather(*[worker(item) for item in pool])
            duration = time.perf_counter() - start_t

        succ_lats = [l for l in latencies if l > 0]
        p50 = percentile(succ_lats, 50)
        p95 = percentile(succ_lats, 95)
        p99 = percentile(succ_lats, 99)
        mean_lat = statistics.mean(succ_lats) if succ_lats else 0
        rps = n_per_level / duration if duration > 0 else 0
        success_rate = (successes / n_per_level) * 100

        sweep_results[f"concurrency_{conc}"] = {
            "concurrency": conc,
            "total_messages": n_per_level,
            "success_count": successes,
            "failure_count": failures,
            "success_rate_pct": round(success_rate, 2),
            "duration_sec": round(duration, 2),
            "throughput_msg_per_sec": round(rps, 2),
            "throughput_msg_per_min": round(rps * 60, 2),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "latency_mean_ms": round(mean_lat, 2),
        }
        
        print(f"    - Throughput   : {round(rps * 60, 2)} pesan/menit")
        print(f"    - Latensi P50  : {round(p50, 2)} ms | P95: {round(p95, 2)} ms | P99: {round(p99, 2)} ms")
        print(f"    - Success Rate : {round(success_rate, 2)}%\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "scalability_sweep_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[+] Hasil Scalability Sweep tersimpan di: {out_file}\n")
    return sweep_results


# =====================================================================
# 2. RELIABILITY & PREDICTION CONSISTENCY TEST
# =====================================================================
async def run_consistency_test(url: str = API_PREDICT_URL, repetitions: int = 20):
    print("=" * 65)
    print(" 2. RELIABILITY & PREDICTION CONSISTENCY TEST")
    print("=" * 65)
    
    test_cases = [
        "dasar bocah bego main gini aja gabisa",
        "halo kawan, terima kasih sudah bantu tadi ya",
        "you are so dumb, uninstall the game already",
        "nice game everyone, play style kalian keren",
    ]
    
    consistency_report = []
    
    async with httpx.AsyncClient() as client:
        for idx, text in enumerate(test_cases, 1):
            print(f"[*] Menguji Sampel Teks #{idx}: '{text}' ({repetitions}x pengulangan)...")
            history = []
            
            for rep in range(repetitions):
                ok, lat, code, raw_resp = await _send_single(client, url, {"text": text}, 10.0)
                if ok:
                    try:
                        data = json.loads(raw_resp)
                        # Ekstrak label dan probabilitas
                        label = data.get("label", data.get("prediction"))
                        confidence = round(data.get("probability", data.get("confidence", 0.0)), 6)
                        history.append({"rep": rep + 1, "label": label, "confidence": confidence, "latency_ms": round(lat, 2)})
                    except Exception as e:
                        history.append({"rep": rep + 1, "error": str(e)})
                else:
                    history.append({"rep": rep + 1, "error": raw_resp})
                await asyncio.sleep(0.02)
                
            labels = [h["label"] for h in history if "label" in h]
            confidences = [h["confidence"] for h in history if "confidence" in h]
            
            unique_labels = set(labels)
            unique_confidences = set(confidences)
            
            is_100_percent_consistent = len(unique_labels) == 1 and len(unique_confidences) == 1
            
            res_item = {
                "text": text,
                "repetitions": repetitions,
                "successful_responses": len(labels),
                "unique_labels_found": list(unique_labels),
                "unique_confidences_found": list(unique_confidences),
                "is_fully_consistent": is_100_percent_consistent,
                "label_consistency_pct": 100.0 if len(unique_labels) == 1 else round((labels.count(labels[0]) / len(labels)) * 100, 2),
                "sample_history": history[:5]  # sertakan 5 sampel pertama
            }
            consistency_report.append(res_item)
            
            status_str = "100% KONSISTEN (DETERMINISTIK)" if is_100_percent_consistent else "VARIASI TERDETEKSI"
            print(f"    - Status Konsistensi : {status_str}")
            print(f"    - Label Terdeteksi   : {unique_labels}")
            print(f"    - Unique Confidence  : {unique_confidences}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "consistency_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(consistency_report, f, indent=2)
    print(f"[+] Hasil Uji Konsistensi tersimpan di: {out_file}\n")
    return consistency_report


# =====================================================================
# 3. FAULT TOLERANCE & MTTR RECOVERY EXPERIMENT
# =====================================================================
def run_cmd_sync(cmd: list[str]):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode == 0, res.stdout.strip(), res.stderr.strip()


async def run_fault_and_recovery_test(container_name: str = "cyberbully_api"):
    print("=" * 65)
    print(f" 3. FAULT TOLERANCE & MTTR RECOVERY TEST (Target Container: {container_name})")
    print("=" * 65)
    
    # Step A: Cek baseline kesehatan sebelum fault injection
    async with httpx.AsyncClient() as client:
        ok_before, lat_before, _, _ = await _send_single(client, API_HEALTH_URL, {}, 5.0, method="GET")
        print(f"[*] Status awal container '{container_name}': {'ONLINE' if ok_before else 'OFFLINE'}")
        
        # Step B: Fault Injection (Stop Container)
        print(f"[*] [FAULT INJECTION] Mematikan container '{container_name}' (docker stop)...")
        stop_success, _, _ = run_cmd_sync(["docker", "stop", container_name])
        if not stop_success:
            print(f"[!] Gagal mematikan container '{container_name}'. Menghentikan pengujian fault.")
            return {}
        
        print(f"[+] Container '{container_name}' berhasil dimatikan.")
        
        # Step C: Uji Respon Sistem Saat Service Down (Fault Tolerance Check)
        print("[*] Mengirim HTTP request ke API saat service OFFLINE untuk menguji penanganan error...")
        ok_during, lat_during, code_during, err_during = await _send_single(client, API_PREDICT_URL, {"text": "test error"}, 3.0, method="POST")
        print(f"    - Respons saat down : Success={ok_during}, HTTP Code={code_during}, Latensi={round(lat_during, 2)}ms")
        print(f"    - Error Detail       : {err_during[:120]}")
        
        # Step D: Trigger Recovery (Docker Start) & Ukur MTTR
        print(f"\n[*] [RECOVERY] Menyalakan kembali container '{container_name}' (docker start) & mengukur MTTR...")
        start_recovery_time = time.perf_counter()
        start_success, _, _ = run_cmd_sync(["docker", "start", container_name])
        
        if not start_success:
            print(f"[!] Gagal menyalakan kembali container '{container_name}'.")
            return {}

        # Polling /health hingga HTTP 200
        mttr_seconds = None
        max_wait = 60.0
        poll_interval = 0.5
        
        while (time.perf_counter() - start_recovery_time) < max_wait:
            ok_poll, _, code_poll, _ = await _send_single(client, API_HEALTH_URL, {}, 2.0, method="GET")
            if ok_poll and code_poll == 200:
                mttr_seconds = time.perf_counter() - start_recovery_time
                break
            await asyncio.sleep(poll_interval)

        if mttr_seconds is not None:
            print(f"[+] Container '{container_name}' kembali ONLINE dan SEHAT!")
            print(f"    - Mean Time to Recovery (MTTR) : {round(mttr_seconds, 2)} detik")
        else:
            print(f"[!] Timeout: Container '{container_name}' tidak kunjung merespons dalam {max_wait} detik.")
            
        recovery_results = {
            "target_container": container_name,
            "fault_injection_applied": True,
            "error_response_during_fault": {
                "success": ok_during,
                "http_code": code_during,
                "error_detail": err_during
            },
            "recovery_successful": mttr_seconds is not None,
            "mttr_seconds": round(mttr_seconds, 2) if mttr_seconds else None,
            "timestamp": datetime.now().isoformat()
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "fault_and_recovery_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(recovery_results, f, indent=2)
    print(f"[+] Hasil Fault & Recovery Test tersimpan di: {out_file}\n")
    return recovery_results


# =====================================================================
# 4. AVAILABILITY SAMPLER
# =====================================================================
async def run_availability_check(duration_seconds: int = 30, interval_seconds: float = 1.0):
    print("=" * 65)
    print(f" 4. AVAILABILITY SAMPLER TEST ({duration_seconds}s Polling, Interval {interval_seconds}s)")
    print("=" * 65)
    
    total_checks = 0
    success_checks = 0
    failed_checks = 0
    latencies = []
    
    start_time = time.perf_counter()
    async with httpx.AsyncClient() as client:
        while (time.perf_counter() - start_time) < duration_seconds:
            total_checks += 1
            ok, lat, code, _ = await _send_single(client, API_HEALTH_URL, {}, 3.0, method="GET")
            latencies.append(lat)
            if ok:
                success_checks += 1
            else:
                failed_checks += 1
            await asyncio.sleep(interval_seconds)
            
    uptime_pct = (success_checks / total_checks * 100) if total_checks > 0 else 0
    avg_lat = statistics.mean(latencies) if latencies else 0
    
    avail_results = {
        "duration_seconds": duration_seconds,
        "polling_interval_seconds": interval_seconds,
        "total_checks": total_checks,
        "successful_checks": success_checks,
        "failed_checks": failed_checks,
        "uptime_percentage": round(uptime_pct, 2),
        "mean_healthcheck_latency_ms": round(avg_lat, 2)
    }
    
    print(f"[*] Total Health Check Samples : {total_checks}")
    print(f"[*] Berhasil / Gagal            : {success_checks} / {failed_checks}")
    print(f"[*] System Uptime Percentage    : {round(uptime_pct, 2)}%")
    print(f"[*] Rata-rata Latensi Health    : {round(avg_lat, 2)} ms\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "availability_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(avail_results, f, indent=2)
    print(f"[+] Hasil Availability Sampler tersimpan di: {out_file}\n")
    return avail_results


# =====================================================================
# MAIN RUNNER
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="System Quality (NFR) Test Suite")
    parser.add_argument("--test", choices=["scalability", "consistency", "fault", "availability", "all"], default="all",
                        help="Pilih modul pengujian yang ingin dijalankan (default: all)")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" MEMULAI PENGUJIAN SYSTEM QUALITY (RELIABILITY, SCALABILITY, FAULT, AVAILABILITY, RECOVERY)")
    print("=" * 65 + "\n")

    if args.test in ["scalability", "all"]:
        asyncio.run(run_scalability_sweep(API_PREDICT_URL, n_per_level=200))

    if args.test in ["consistency", "all"]:
        asyncio.run(run_consistency_test(API_PREDICT_URL, repetitions=20))

    if args.test in ["fault", "all"]:
        asyncio.run(run_fault_and_recovery_test(container_name="cyberbully_api"))

    if args.test in ["availability", "all"]:
        asyncio.run(run_availability_check(duration_seconds=30, interval_seconds=1.0))

    print("=" * 65)
    print(" SELURUH PENGUJIAN SYSTEM QUALITY SELESAI!")
    print(f" Folder Hasil: {OUTPUT_DIR.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
