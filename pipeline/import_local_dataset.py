import csv
import json
import shutil
import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
METADATA_PATH = RAW_DIR / "metadata.json"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"
TEST_ANCHORAGE_DIR = RAW_DIR / "test_anchorage"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    csv_path = Path(parser.parse_args().csv_file)
    
    if not csv_path.exists():
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
        
    if not records:
        return

    processed, stats, now = [], {}, datetime.utcnow()
    iso_time = now.isoformat() + "Z"
    
    for r in records:
        uid = r.get("user_id", "unknown")
        stats.setdefault(uid, {"count": 0, "images": 0})["count"] += 1
        
        local_img = None
        if src := r.get("image_filename"):
            clean_uid = Path(uid).name.replace("..", "_").replace("/", "_").replace("\\", "_") or "unknown"
            (user_dir := IMAGES_DIR / clean_uid).mkdir(parents=True, exist_ok=True)
            
            src_path = TEST_ANCHORAGE_DIR / src
            if src_path.exists():
                dst = user_dir / f"img_{int(r.get('id', 0)):03d}{src_path.suffix or '.jpg'}"
                shutil.copy2(src_path, dst)
                local_img = f"{clean_uid}/{dst.name}"
                stats[uid]["images"] += 1

        processed.append({
            "id": int(r.get("id", 0)), "user_id": uid, "caption": r.get("caption", ""),
            "timestamp": r.get("date", ""), "lat": float(r.get("latitude", 0)),
            "lon": float(r.get("longitude", 0)), "image_url": r.get("image_url", ""),
            "local_image": local_img, "snapshot_id": now.strftime("snapshot_%Y_%m_%d"),
            "created_at": iso_time
        })

    snapshot = {"snapshot_id": processed[0]["snapshot_id"], "created_at": iso_time, "record_count": len(processed), "user_count": len(stats), "user_stats": stats, "records": processed}
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    conn = init_db()
    for t in ("emotion_scores", "temporal_features", "location_descriptions", "memories"):
        conn.execute(f"DELETE FROM {t}")
        
    for r in processed:
        conn.execute("INSERT OR REPLACE INTO memories (id, user_id, caption, timestamp, lat, lon, image_url, local_image, snapshot_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (r["id"], r["user_id"], r["caption"], r["timestamp"], r["lat"], r["lon"], r["image_url"], r["local_image"], r["snapshot_id"], r["created_at"]))
    conn.commit()
    conn.close()

    snap_dir = SNAPSHOTS_DIR / (snapshot["snapshot_id"] if not (SNAPSHOTS_DIR / snapshot["snapshot_id"]).exists() else f"{snapshot['snapshot_id']}_{now.strftime('%H%M%S')}")
    snap_dir.mkdir(parents=True, exist_ok=True)
    
    if IMAGES_DIR.exists():
        shutil.copytree(IMAGES_DIR, snap_dir / "images", dirs_exist_ok=True)
        
    with open(snap_dir / "metadata.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

if __name__ == "__main__":
    main()
