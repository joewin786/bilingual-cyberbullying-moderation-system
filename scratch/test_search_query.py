import requests
import re

query = "klarifikasi+dibully"
url = f"https://www.youtube.com/results?search_query={query}"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

r = requests.get(url, headers=headers)
video_ids = list(set(re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', r.text)))
print(f"Status: {r.status_code}, Found: {len(video_ids)} video IDs: {video_ids[:5]}")
