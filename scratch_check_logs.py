import sqlite3
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = sqlite3.connect("discord_bot/violations.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM action_logs ORDER BY id DESC LIMIT 10").fetchall()
for r in rows:
    print(f"ID: {r['id']} | Content: {repr(r['message_content'])} | Action: {repr(r['action'])} | Conf: {r['confidence']} | Tier: {r['action_tier']} | Severity: {r['severity']} | TS: {r['timestamp']}")
conn.close()
