#!/usr/bin/env python3
"""
Alternative Phase 1: Local Data Import Script
Imports records from a local CSV file, copies images from a local directory,
and creates a frozen snapshot for experiments into the SQLite database.
"""

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
METADATA_PATH = RAW_DIR / "metadata.json"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"
TEST_ANCHORAGE_DIR = RAW_DIR / "test_anchorage"

def import_records_from_csv(csv_path: str) -> list:
    """Read records from CSV."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def copy_image(source_filename: str, user_id: str, record_id: int) -> str | None:
    """Copy image from test_anchorage to user directory."""
    if not source_filename:
        return None
        
    sanitized_user_id = Path(user_id).name.replace("..", "_").replace("/", "_").replace("\\", "_")
    if not sanitized_user_id:
        sanitized_user_id = "unknown"
        
    user_dir = IMAGES_DIR / sanitized_user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = TEST_ANCHORAGE_DIR / source_filename
    if not source_path.exists():
        print(f"   [WARN] Source image not found: {source_path}")
        return None
        
    ext = source_path.suffix or ".jpg"
    filename = f"img_{int(record_id):03d}{ext}"
    filepath = user_dir / filename
    
    shutil.copy2(source_path, filepath)
    
    relative_path = f"{sanitized_user_id}/{filename}"
    print(f"   [INFO] Copied: {relative_path}")
    return relative_path


def process_records(records: list) -> tuple[list, dict]:
    processed_records = []
    user_stats = {}
    
    now_iso = datetime.utcnow().isoformat() + "Z"
    
    for record in records:
        user_id = record.get("user_id", "unknown")
        record_id = int(record.get("id", 0))
        image_filename = record.get("image_filename", "")
        
        if user_id not in user_stats:
            user_stats[user_id] = {"count": 0, "images": 0}
        user_stats[user_id]["count"] += 1
        
        local_image = copy_image(image_filename, user_id, record_id)
        if local_image:
            user_stats[user_id]["images"] += 1
            
        processed_record = {
            "id": record_id,
            "user_id": user_id,
            "caption": record.get("caption", ""),
            "timestamp": record.get("date", ""),
            "lat": float(record.get("latitude", 0)),
            "lon": float(record.get("longitude", 0)),
            "image_url": record.get("image_url", ""),
            "local_image": local_image,
            "processed": 0,
            "processing_version": 1,
            "created_at": now_iso
        }
        processed_records.append(processed_record)
        
    return processed_records, user_stats

def create_snapshot(records: list, user_stats: dict) -> dict:
    now = datetime.utcnow()
    snapshot_id = now.strftime("snapshot_%Y_%m_%d")
    return {
        "snapshot_id": snapshot_id,
        "created_at": now.isoformat() + "Z",
        "record_count": len(records),
        "user_count": len(user_stats),
        "user_stats": user_stats,
        "records": records
    }

def save_metadata(snapshot: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"\n[INFO] Metadata saved: {METADATA_PATH}")

def freeze_snapshot(snapshot: dict) -> Path:
    snapshot_dir = SNAPSHOTS_DIR / snapshot["snapshot_id"]
    if snapshot_dir.exists():
        suffix = datetime.utcnow().strftime("_%H%M%S")
        snapshot_dir = SNAPSHOTS_DIR / (snapshot["snapshot_id"] + suffix)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    if IMAGES_DIR.exists():
        shutil.copytree(IMAGES_DIR, snapshot_dir / "images", dirs_exist_ok=True)
        
    snapshot_metadata_path = snapshot_dir / "metadata.json"
    with open(snapshot_metadata_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
        
    print(f"[INFO] Snapshot frozen: {snapshot_dir}")
    return snapshot_dir

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import local CSV dataset into the pipeline.")
    parser.add_argument("csv_file", help="Path to the CSV file to import")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
        
    print("=" * 60)
    print("DREAMS Research - Local Data Import")
    print("=" * 60)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Step 1: Reading records from CSV...")
    records = import_records_from_csv(csv_path)
    if not records:
        print("\n[WARN] No records to process. Exiting.")
        return
        
    print(f"\n[INFO] Step 2: Processing {len(records)} records and copying images from test_anchorage...")
    processed_records, user_stats = process_records(records)
    
    print("\n[INFO] Step 3: Creating snapshot...")
    snapshot = create_snapshot(processed_records, user_stats)
    
    save_metadata(snapshot)
    
    print("\n[INFO] Step 5: Inserting records into SQLite...")
    conn = init_db()
    conn.execute("DELETE FROM emotion_scores")
    conn.execute("DELETE FROM temporal_features")
    conn.execute("DELETE FROM location_descriptions")
    conn.execute("DELETE FROM memories")
    for rec in processed_records:
        conn.execute(
            '''INSERT OR REPLACE INTO memories
               (id, user_id, caption, timestamp, lat, lon,
                image_url, local_image, snapshot_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                rec["id"], rec["user_id"], rec["caption"],
                rec["timestamp"], rec["lat"], rec["lon"],
                rec["image_url"], rec["local_image"],
                snapshot["snapshot_id"], rec["created_at"],
            ),
        )
    conn.commit()
    print(f"   Inserted {len(processed_records)} records into memories table")
    conn.close()
    
    print("\n[INFO] Step 6: Freezing snapshot...")
    snapshot_path = freeze_snapshot(snapshot)
    
    print("\n" + "=" * 60)
    print("[OK] Import Complete!")
    print("=" * 60)
    print(f"   [INFO] Records: {snapshot['record_count']}")
    print(f"   [INFO] Snapshot: {snapshot_path}")

if __name__ == "__main__":
    main()
