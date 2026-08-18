"""
scrape_youtube.py
=================
Script untuk melakukan scraping komentar YouTube berdasarkan Channel Handle / Username.
Script akan otomatis mencari daftar video terbaru dari channel tersebut dan mendownload komentarnya.

Library yang digunakan: youtube-comment-downloader (ringan, tanpa perlu API Key)
"""

import os
import sys
import re
import csv
import requests
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

# Otorun install library jika belum ada
try:
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT
except ImportError:
    print("Library 'youtube-comment-downloader' belum terinstall. Menginstall sekarang...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-comment-downloader"])
    from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

# ──────────────────────────────────────────────────────────────
# Konfigurasi Target Scraping
# ──────────────────────────────────────────────────────────────
TARGET_QUERIES = [
    "dihujat netizen",
    "klarifikasi dibully",
    "korban cyberbullying",
    "bully netizen",
    "kasus perundungan"
]

VIDEOS_PER_QUERY = 5    # Ambil 5 video teratas dari setiap kueri pencarian
LIMIT_PER_VIDEO = 300   # Periksa hingga 300 komentar terbaru per video
MIN_CHAR_LENGTH = 5     # Batasi panjang karakter minimal komentar
OUTPUT_PATH = Path("data/processed/youtube_scraped.csv")

# Daftar kata kunci pemicu / indikasi komentar toxic / cyberbullying personal (Indo)
TOXIC_KEYWORDS = {
    # Kata kasar/makian kotor
    "anjing", "anjg", "anj", "bangsat", "goblok", "tolol", "bego", "dungu", "babi", "asu", 
    "kampret", "idiot", "sinting", "gila", "najis", "bacot", "otak", "freak", "cringe", "bocil",
    # Body shaming / Fisik
    "gendut", "gemuk", "kurus", "peot", "dekil", "kucel", "hitam", "jelek", "burik", "tua", 
    "keriput", "oplas", "operasian", "palsu", "dempul", "makeup", "muka", "wajah", "fisik",
    # Serangan karakter & Moral / Drama
    "gatel", "murahan", "pelakor", "selingkuh", "jalang", "lonte", "sok", "caper", "pansos", 
    "benci", "sampah", "cacad", "cacat", "gatel", "murahan", "pelacur", "pamer", "munafik", 
    "hujat", "dihujat", "nyinyir", "karma", "palsu", "bohong", "drama", "settingan", "gimmick",
    "lebay", "alay", "norak", "kampungan", "najis"
}

def is_toxic_candidate(text: str) -> bool:
    """Cek apakah komentar mengandung salah satu kata kunci pemicu toxic."""
    text_lower = text.lower()
    for word in TOXIC_KEYWORDS:
        if word in text_lower:
            return True
    return False

# Headers global untuk bypass block bot YouTube
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

def get_videos_by_query(query: str, limit: int) -> list:
    """Mencari daftar video ID dari hasil pencarian YouTube berdasarkan kata kunci/query."""
    query_clean = query.strip().replace(" ", "+")
    url = f"https://www.youtube.com/results?search_query={query_clean}"
    video_ids = []
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  [Warning] Gagal mencari video (status code: {r.status_code})")
            return []
            
        found = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text)
        
        seen = set()
        for vid in found:
            if vid not in seen:
                seen.add(vid)
                video_ids.append({
                    "id": vid,
                    "title": f"Video pencarian {query}"
                })
                if len(video_ids) >= limit:
                    break
                    
    except Exception as e:
        print(f"  [Error] Gagal mengambil list video berdasarkan query: {e}")
        
    return video_ids

def clean_text(text: str) -> str:
    """Pembersihan teks sederhana (hapus newline berlebih, whitespace ganda)."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def main():
    print("=" * 60)
    print("YOUTUBE SEARCH QUERY COMMENT SCRAPER")
    print("=" * 60)

    downloader = YoutubeCommentDownloader()
    scraped_data = []
    seen_texts = set()

    for query in TARGET_QUERIES:
        print(f"\n[*] Memproses query pencarian: \"{query}\" ...")
        
        videos = get_videos_by_query(query, VIDEOS_PER_QUERY)
        print(f"  -> Berhasil mendeteksi {len(videos)} video dari pencarian.")
        
        for idx, video in enumerate(videos):
            video_id = video["id"]
            video_title = video["title"]
            print(f"\n  [{idx+1}/{len(videos)}] Scraping video ID: {video_id}")
            
            try:
                generator = downloader.get_comments(video_id, sort_by=SORT_BY_RECENT)
                count = 0
                
                for comment in generator:
                    if count >= LIMIT_PER_VIDEO:
                        break

                    text = clean_text(comment.get("text", ""))
                    
                    if len(text) < MIN_CHAR_LENGTH:
                        continue
                    
                    if not is_toxic_candidate(text):
                        continue
                    
                    if text.lower() in seen_texts:
                        continue
                    
                    seen_texts.add(text.lower())
                    
                    scraped_data.append({
                        "text": text,
                        "label": ""  # Kosong untuk dilabeli secara manual
                    })
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"    -> Berhasil mengambil {count} komentar...")

                print(f"    [Success] Selesai! Berhasil mengumpulkan {count} komentar.")
                
            except Exception as e:
                print(f"    [Error] Gagal mengambil komentar untuk video {video_id}: {e}")

    # Simpan hasil scraping ke CSV
    if not scraped_data:
        print("\n[Warning] Tidak ada data komentar yang berhasil dikumpulkan.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    mode = "w"
    write_header = True
    if OUTPUT_PATH.exists():
        print(f"\nFile {OUTPUT_PATH.name} sudah ada.")
        mode = "a"
        write_header = False
        print("  -> Menggabungkan (append) data baru ke file yang sudah ada.")

    with open(OUTPUT_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        if write_header:
            writer.writeheader()
        writer.writerows(scraped_data)

    print(f"\n{'='*60}")
    print(f"SCRAPING SELESAI!")
    print(f"Total komentar baru didapat : {len(scraped_data)}")
    print(f"Disimpan di                 : {OUTPUT_PATH.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
