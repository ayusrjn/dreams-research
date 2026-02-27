import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH
from db import init_db

def parse(ts):
    if not ts: return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try: return datetime.strptime(ts, fmt)
        except ValueError: pass
    return None

def main():
    if not RAW_METADATA_PATH.exists():
        sys.exit(1)
        
    with open(RAW_METADATA_PATH) as f:
        records = json.load(f).get("records", [])

    first_entries = {}
    for r in records:
        if (uid := r.get("user_id")) and (dt := parse(r.get("timestamp"))) and (uid not in first_entries or dt < first_entries[uid]):
            first_entries[uid] = dt

    results = []
    for r in records:
        if not (dt := parse(r.get("timestamp"))):
            continue
            
        rad = 2 * math.pi * dt.hour / 24
        results.append({
            "id": r["id"],
            "user_id": r.get("user_id"),
            "absolute_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "relative_day": (dt - first_entries[r.get("user_id")]).days if r.get("user_id") in first_entries else 0,
            "sin_hour": round(math.sin(rad), 4),
            "cos_hour": round(math.cos(rad), 4),
        })

    if results:
        conn = init_db()
        for r in results:
            conn.execute(
                "INSERT OR REPLACE INTO temporal_features (id, user_id, absolute_utc, relative_day, sin_hour, cos_hour) VALUES (?, ?, ?, ?, ?, ?)",
                (r["id"], r["user_id"], r["absolute_utc"], r["relative_day"], r["sin_hour"], r["cos_hour"])
            )
        conn.commit()
        conn.close()

if __name__ == "__main__":
    main()
