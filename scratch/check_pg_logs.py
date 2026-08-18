import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv(project_root / ".env")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "cyberbully_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

def check_logs():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    print("=== SEARCH FOR 'dataset' IN ACTION LOGS ===")
    cur.execute("SELECT * FROM action_logs WHERE message_content LIKE '%dataset%' OR message_content LIKE '%detect%' ORDER BY id DESC")
    for r in cur.fetchall():
        print(f"ID: {r['id']} | Content: {repr(r['message_content'])} | Action: {r['action']} | Conf: {r['confidence']:.3f} | Tier: {r['action_tier']} | Severity: {r['severity']} | TS: {r['timestamp']}")
        
    print("\n=== SEARCH FOR 'dataset' IN REVIEW QUEUE ===")
    cur.execute("SELECT * FROM mod_review_queue WHERE message_content LIKE '%dataset%' OR message_content LIKE '%detect%' ORDER BY id DESC")
    for r in cur.fetchall():
        print(f"ID: {r['id']} | Content: {repr(r['message_content'])} | Status: {r['status']} | Conf: {r['confidence']:.3f} | Severity: {r['severity']} | TS: {r['timestamp']}")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_logs()
