import requests
import re
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_RECENT

queries = ["dihujat+netizen", "klarifikasi+dibully"]
downloader = YoutubeCommentDownloader()
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

TOXIC_KEYWORDS = {
    "anjing", "anjg", "anj", "bangsat", "goblok", "tolol", "bego", "dungu", "babi", "asu", 
    "kampret", "idiot", "sinting", "gila", "najis", "bacot", "otak", "freak", "cringe", "bocil",
    "gendut", "gemuk", "kurus", "peot", "dekil", "kucel", "hitam", "jelek", "burik", "tua", 
    "keriput", "oplas", "operasian", "palsu", "dempul", "makeup", "muka", "wajah", "fisik",
    "gatel", "murahan", "pelakor", "selingkuh", "jalang", "lonte", "sok", "caper", "pansos", 
    "benci", "sampah", "cacad", "cacat", "pelacur", "pamer", "munafik", 
    "hujat", "dihujat", "nyinyir", "karma", "bohong", "drama", "settingan", "gimmick",
    "lebay", "alay", "norak", "kampungan", "najis"
}

def is_toxic(text):
    t = text.lower()
    return any(w in t for w in TOXIC_KEYWORDS)

vids = []
for q in queries:
    url = f"https://www.youtube.com/results?search_query={q}"
    r = requests.get(url, headers=headers)
    vids.extend(re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text))

vids = list(set(vids))
print(f"Found {len(vids)} unique videos: {vids[:5]}")

collected = []
for vid in vids[:3]:
    print(f"Scraping video {vid}...")
    try:
        gen = downloader.get_comments(vid, sort_by=SORT_BY_RECENT)
        count = 0
        for comment in gen:
            if count >= 100:
                break
            txt = comment.get("text", "")
            if len(txt) > 5 and is_toxic(txt):
                collected.append(txt)
                count += 1
    except Exception as e:
        print(f"Error: {e}")

print(f"Collected {len(collected)} filtered comments.")
for idx, c in enumerate(collected[:10]):
    print(f"{idx+1}. {c[:120]}")
