#!/usr/bin/env python3
"""
Phase 1: Data Pull & Freezing Script
Pulls unprocessed records from Cloudflare D1 Database, downloads images
organized by user_id, and creates a frozen snapshot for experiments.

Output Structure:
    data/raw/
        images/
            user_01/
                img_001.jpg
            user_02/
                ...
        metadata.json
    
    data/snapshots/
        snapshot_YYYY_MM_DD/
            (copy of frozen data)
"""

import os
import json
import shutil
import sys
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
from db import init_db


# Configuration - Update these with your actual values
D1_API_TOKEN = os.getenv("D1_API_TOKEN", "")
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
D1_DATABASE_ID = os.getenv("D1_DATABASE_ID", "")

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
IMAGES_DIR = RAW_DIR / "images"
METADATA_PATH = RAW_DIR / "metadata.json"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"


def pull_records_from_d1() -> list:
    """
    Pull unprocessed records from Cloudflare D1 Database.
    
    Returns:
        List of record dictionaries from the database.
    """
    if not all([D1_API_TOKEN, CF_ACCOUNT_ID, D1_DATABASE_ID]):
        print("[WARN] D1 credentials not configured. Using empty dataset.")
        print("   Set environment variables: D1_API_TOKEN, CF_ACCOUNT_ID, D1_DATABASE_ID")
        return []
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/d1/database/{D1_DATABASE_ID}/query"
    
    headers = {
        "Authorization": f"Bearer {D1_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Query for unprocessed records
    payload = {
        "sql": "SELECT id, user_id, caption, timestamp, lat, lon, image_url, processed, processing_version, created_at FROM memories WHERE processed = 0 OR processed IS NULL"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            results = data.get("result", [{}])[0].get("results", [])
            print(f"[OK] Pulled {len(results)} records from D1")
            return results
        else:
            print(f"[ERROR] D1 API error: {data.get('errors', 'Unknown error')}")
            return []
            
    except requests.RequestException as e:
        print(f"[ERROR] Network error: {e}")
        return []


def download_image(url: str, user_id: str, record_id: int) -> str | None:
    """
    Download an image from Cloudinary and save to user-specific folder.
    
    Args:
        url: Cloudinary image URL
        user_id: User ID for folder organization
        record_id: Record ID for filename
        
    Returns:
        Relative path from images dir if successful, None otherwise.
    """
    if not url:
        return None
    
    # Sanitize user_id to prevent path traversal attacks
    sanitized_user_id = Path(user_id).name.replace("..", "_").replace("/", "_").replace("\\", "_")
    if not sanitized_user_id:
        sanitized_user_id = "unknown"
    
    # Create user directory
    user_dir = IMAGES_DIR / sanitized_user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Extract extension from URL
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or ".jpg"
        
        filename = f"img_{record_id:03d}{ext}"
        filepath = user_dir / filename
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        relative_path = f"{sanitized_user_id}/{filename}"
        print(f"   [INFO] Downloaded: {relative_path}")
        return relative_path
        
    except requests.RequestException as e:
        print(f"   [WARN] Failed to download image for record {record_id}: {e}")
        return None


def process_records(records: list) -> tuple[list, dict]:
    """
    Process records and download images organized by user.
    
    Args:
        records: List of raw records from D1
        
    Returns:
        Tuple of (processed_records, user_stats)
    """
    processed_records = []
    user_stats = {}
    
    for record in records:
        user_id = record.get("user_id", "unknown")
        record_id = record.get("id", 0)
        image_url = record.get("image_url", "")
        
        # Track user stats
        if user_id not in user_stats:
            user_stats[user_id] = {"count": 0, "images": 0}
        user_stats[user_id]["count"] += 1
        
        # Download image to user folder
        local_image = download_image(image_url, user_id, record_id)
        if local_image:
            user_stats[user_id]["images"] += 1
        
        # Build processed record matching D1 schema
        processed_record = {
            "id": record_id,
            "user_id": user_id,
            "caption": record.get("caption", ""),
            "timestamp": record.get("timestamp", ""),
            "lat": record.get("lat"),
            "lon": record.get("lon"),
            "image_url": image_url,
            "local_image": local_image,
            "processed": record.get("processed", 0),
            "processing_version": record.get("processing_version"),
            "created_at": record.get("created_at", "")
        }
        processed_records.append(processed_record)
    
    return processed_records, user_stats


def create_snapshot(records: list, user_stats: dict) -> dict:
    """
    Create a frozen snapshot metadata object.
    
    Args:
        records: Processed records
        user_stats: Statistics per user
        
    Returns:
        Snapshot metadata dictionary
    """
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
    """Save snapshot metadata to JSON file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(METADATA_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    
    print(f"\n[INFO] Metadata saved: {METADATA_PATH}")


def freeze_snapshot(snapshot: dict) -> Path:
    """
    Create a frozen copy of the data in snapshots directory.
    
    Args:
        snapshot: Snapshot metadata
        
    Returns:
        Path to the frozen snapshot directory
    """
    snapshot_dir = SNAPSHOTS_DIR / snapshot["snapshot_id"]
    
    if snapshot_dir.exists():
        # Add timestamp suffix if snapshot already exists
        suffix = datetime.utcnow().strftime("_%H%M%S")
        snapshot_dir = SNAPSHOTS_DIR / (snapshot["snapshot_id"] + suffix)
    
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    if IMAGES_DIR.exists():
        shutil.copytree(IMAGES_DIR, snapshot_dir / "images", dirs_exist_ok=True)
    
    # Copy metadata
    snapshot_metadata_path = snapshot_dir / "metadata.json"
    with open(snapshot_metadata_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    
    print(f"[INFO] Snapshot frozen: {snapshot_dir}")
    return snapshot_dir


def main():
    """Main execution flow for Phase 1."""
    print("=" * 60)
    print("DREAMS Research - Phase 1: Data Pull & Freezing")
    print("=" * 60)
    print()
    
    # Ensure directories exist
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Pull records from D1
    print("[INFO] Step 1: Pulling records from D1...")
    records = pull_records_from_d1()
    
    if not records:
        print("\n[WARN] No records to process. Exiting.")
        return
    
    # Step 2: Process records and download images
    print(f"\n[INFO] Step 2: Processing {len(records)} records and downloading images...")
    processed_records, user_stats = process_records(records)
    
    # Step 3: Create snapshot
    print("\n[INFO] Step 3: Creating snapshot...")
    snapshot = create_snapshot(processed_records, user_stats)
    
    # Step 4: Save metadata (raw JSON preserved for snapshot freezing)
    save_metadata(snapshot)
    
    # Step 5: Insert into SQLite
    print("\n[INFO] Step 5: Inserting records into SQLite...")
    conn = init_db()
    conn.execute("DELETE FROM memories")  # Fresh pull replaces all
    for rec in processed_records:
        conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, user_id, caption, timestamp, lat, lon,
                image_url, local_image, snapshot_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    
    # Step 6: Freeze snapshot
    print("\n[INFO] Step 6: Freezing snapshot...")
    snapshot_path = freeze_snapshot(snapshot)
    
    # Summary
    print("\n" + "=" * 60)
    print("[OK] Phase 1 Complete!")
    print("=" * 60)
    print(f"   [INFO] Records: {snapshot['record_count']}")
    print(f"   [INFO] Users: {snapshot['user_count']}")
    print(f"   [INFO] Raw data: {RAW_DIR}")
    print(f"   [INFO] Database: dreams.db")
    print(f"   [INFO] Snapshot: {snapshot_path}")
    print("\nThis snapshot is the **experiment boundary**.")


if __name__ == "__main__":
    main()
