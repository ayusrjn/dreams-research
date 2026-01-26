#!/usr/bin/env python3
"""
Phase 2D: Temporal Feature Extraction

Transforms raw timestamps into research-ready temporal features:
1. Absolute timestamp (ISO-8601 UTC)
2. Circadian coordinates (sin/cos of hour for cyclic time-of-day)
3. Relative epoch (days since user's first entry)

Output:
    data/processed/
        temporal_features.csv - Temporal features per record
"""

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    PROCESSED_DIR,
    TEMPORAL_FEATURES_PATH,
)


def load_metadata() -> dict:
    """Load the frozen snapshot metadata."""
    if not RAW_METADATA_PATH.exists():
        print(f"❌ Metadata not found: {RAW_METADATA_PATH}")
        print("   Run Phase 1 first: python pipeline/pull_data.py")
        sys.exit(1)
    
    with open(RAW_METADATA_PATH) as f:
        return json.load(f)


def parse_timestamp(timestamp_str: str) -> datetime | None:
    """
    Parse ISO-8601 timestamp string to datetime object.
    
    Handles various formats:
        - 2026-01-26T14:30:00Z
        - 2026-01-26T14:30:00.000Z
        - 2026-01-26 14:30:00
    
    Returns:
        datetime object in UTC, or None if parsing fails.
    """
    if not timestamp_str:
        return None
    
    # Common ISO-8601 formats
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    return None


def compute_circadian_coords(hour: int) -> tuple[float, float]:
    """
    Compute cyclic time-of-day coordinates.
    
    Transforms hour (0-23) to a point on the unit circle:
        sin_hour = sin(2π × hour / 24)
        cos_hour = cos(2π × hour / 24)
    
    This ensures that 23:00 and 01:00 are mathematically close,
    allowing "Late Night" to be identified as a singular context.
    
    Args:
        hour: Hour of day (0-23)
        
    Returns:
        Tuple of (sin_hour, cos_hour)
    """
    radians = 2 * math.pi * hour / 24
    sin_hour = math.sin(radians)
    cos_hour = math.cos(radians)
    return round(sin_hour, 4), round(cos_hour, 4)


def compute_user_first_entries(records: list) -> dict[str, datetime]:
    """
    Find the first entry timestamp for each user.
    
    Args:
        records: List of record dictionaries
        
    Returns:
        Dictionary mapping user_id to their first entry datetime
    """
    user_first = {}
    
    for record in records:
        user_id = record.get("user_id")
        timestamp_str = record.get("timestamp")
        
        if not user_id or not timestamp_str:
            continue
        
        dt = parse_timestamp(timestamp_str)
        if dt is None:
            continue
        
        if user_id not in user_first or dt < user_first[user_id]:
            user_first[user_id] = dt
    
    return user_first


def extract_temporal_features(metadata: dict) -> list[dict]:
    """
    Extract temporal features for all records.
    
    Returns:
        List of dictionaries with temporal features per record.
    """
    records = metadata.get("records", [])
    results = []
    
    print(f"📅 Processing {len(records)} records...")
    
    # Step 1: Compute first entry per user (for relative epoch)
    print("   Computing user first entries...")
    user_first = compute_user_first_entries(records)
    print(f"   Found {len(user_first)} users with valid timestamps")
    
    # Step 2: Extract features for each record
    for record in records:
        record_id = record.get("id")
        user_id = record.get("user_id")
        timestamp_str = record.get("timestamp")
        
        if not timestamp_str:
            print(f"   ⚠️  Record {record_id}: No timestamp (skipped)")
            continue
        
        dt = parse_timestamp(timestamp_str)
        if dt is None:
            print(f"   ⚠️  Record {record_id}: Invalid timestamp format (skipped)")
            continue
        
        # Circadian coordinates (using UTC hour)
        hour = dt.hour
        sin_hour, cos_hour = compute_circadian_coords(hour)
        
        # Relative epoch (days since user's first entry)
        first_entry = user_first.get(user_id)
        if first_entry:
            relative_day = (dt - first_entry).days
        else:
            relative_day = 0
        
        result = {
            "id": record_id,
            "user_id": user_id,
            "absolute_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "relative_day": relative_day,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
        }
        results.append(result)
        
        print(f"   ✅ Record {record_id}: Day {relative_day}, Hour {hour} → sin={sin_hour:.3f}, cos={cos_hour:.3f}")
    
    return results


def save_outputs(results: list[dict]) -> None:
    """Save temporal features to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = ["id", "user_id", "absolute_utc", "relative_day", "sin_hour", "cos_hour"]
    
    with open(TEMPORAL_FEATURES_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([r.get(h, "") for h in headers])
    
    print(f"💾 Temporal features saved: {TEMPORAL_FEATURES_PATH}")
    print(f"   Records: {len(results)}")


def main():
    """Main execution flow for Phase 2D: Temporal Features."""
    print("=" * 60)
    print("DREAMS Research - Phase 2D: Temporal Feature Extraction")
    print("=" * 60)
    print()
    
    # Step 1: Load metadata
    print("📂 Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")
    
    # Step 2: Extract temporal features
    print("\n🔍 Step 2: Extracting temporal features...")
    results = extract_temporal_features(metadata)
    
    if not results:
        print("\n⚠️  No temporal features extracted. Check your timestamps.")
        return
    
    # Step 3: Save outputs
    print("\n💾 Step 3: Saving outputs...")
    save_outputs(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 2D Complete!")
    print("=" * 60)
    print(f"   📊 Processed: {len(results)} records")
    print(f"   📁 Output: {TEMPORAL_FEATURES_PATH}")
    
    # Validation info
    if results:
        days = [r["relative_day"] for r in results]
        print(f"\n📈 Temporal Range:")
        print(f"   Relative days: {min(days)} to {max(days)}")


if __name__ == "__main__":
    main()
