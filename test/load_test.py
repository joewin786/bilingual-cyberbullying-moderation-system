import asyncio
import aiohttp
import time
import argparse
import numpy as np
import sys

API_URL = "http://localhost:8000/predict"

# Diverse test pool of texts representing normal and bullying comments in Indonesia
TEST_TEXTS = [
    # Non-bully
    "semangat ya untuk hari ini!",
    "terima kasih atas bantuannya kak, keren banget.",
    "bagus sekali permainannya tadi, selamat ya!",
    "apakah besok kita ada rapat kelompok?",
    "halo selamat pagi dunia, semoga hari ini menyenangkan.",
    "makasih banyak ya infonya, sangat membantu.",
    "wah hebat sekali, teruskan prestasinya!",
    "menurutku desainnya sudah rapi dan elegan.",
    "siap kak, nanti saya kirim berkasnya ya.",
    "lucu banget videonya, bikin ketawa terus wkwk.",
    # Bully
    "dasar kamu bodoh gila goblok banget sih!",
    "muka jelek kayak babi masih aja sok eksis.",
    "anjing lu bangsat, mati aja sana gak berguna hidup lu.",
    "tolol banget mainnya, mending uninstall aja gamenya bego.",
    "dasar idiot, penjelasan gampang gini aja gak paham-paham.",
    "bajingan lu ya, awas aja nanti gw bales lu.",
    "mental lemah cupu banget gitu aja nangis.",
    "sampah masyarakat lu, gak pantes ada di sini.",
    "lu emang manusia paling ga guna sedunia bego.",
    "bacot lu kontol, gausah banyak omong deh."
]

async def send_request(session, text, request_times):
    payload = {"text": text, "lang": "id"}
    t0 = time.perf_counter()
    try:
        async with session.post(API_URL, json=payload, timeout=10) as response:
            latency = (time.perf_counter() - t0) * 1000  # ms
            status = response.status
            if status == 200:
                request_times.append(latency)
                return True
            else:
                return False
    except Exception as e:
        return False

async def worker(session, queue, request_times):
    while True:
        text = await queue.get()
        if text is None:
            queue.task_done()
            break
        await send_request(session, text, request_times)
        queue.task_done()

async def run_load_test(concurrency, total_requests):
    print(f"[*] Starting load test...")
    print(f"    Concurrency    : {concurrency} workers")
    print(f"    Total Requests : {total_requests} requests")
    
    # Fill queue
    queue = asyncio.Queue()
    for i in range(total_requests):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        await queue.put(text)
        
    # Add termination signals for workers
    for _ in range(concurrency):
        await queue.put(None)
        
    request_times = []
    
    t_start = time.perf_counter()
    
    # Start session and workers
    async with aiohttp.ClientSession() as session:
        workers = [
            asyncio.create_task(worker(session, queue, request_times))
            for _ in range(concurrency)
        ]
        
        await queue.join()
        await asyncio.gather(*workers)
        
    t_end = time.perf_counter()
    total_time = t_end - t_start
    
    success_count = len(request_times)
    fail_count = total_requests - success_count
    
    print("\n" + "=" * 60)
    print("CYBERBULLYING DETECTION API — LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Duration               : {total_time:.2f} seconds")
    print(f"Total Requests Sent    : {total_requests}")
    print(f"Successful Requests    : {success_count} ({success_count/total_requests*100:.1f}%)")
    print(f"Failed Requests        : {fail_count} ({fail_count/total_requests*100:.1f}%)")
    
    if success_count > 0:
        throughput = success_count / total_time
        avg_lat = np.mean(request_times)
        min_lat = np.min(request_times)
        max_lat = np.max(request_times)
        p50 = np.percentile(request_times, 50)
        p95 = np.percentile(request_times, 95)
        p99 = np.percentile(request_times, 99)
        
        print(f"Throughput (RPS)       : {throughput:.2f} req/sec")
        print(f"Average Latency        : {avg_lat:.2f} ms")
        print(f"Min Latency            : {min_lat:.2f} ms")
        print(f"Max Latency            : {max_lat:.2f} ms")
        print(f"P50 Latency (Median)   : {p50:.2f} ms")
        print(f"P95 Latency            : {p95:.2f} ms")
        print(f"P99 Latency            : {p99:.2f} ms")
    else:
        print("Error: No successful requests recorded.")
        
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Load testing tool for Cyberbullying Detection API")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent connections")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
    args = parser.parse_args()
    
    # Check Python version and event loop policy for Windows if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_load_test(args.concurrency, args.requests))

if __name__ == "__main__":
    main()
