#!/usr/bin/env python3
"""
Grand Fusion: Master Manifest Verification & Export

The master_manifest is now a SQL VIEW that automatically joins all tables.
This script verifies integrity and optionally exports to Parquet.

Usage:
    python pipeline/create_master_manifest.py                # verify only
    python pipeline/create_master_manifest.py --export       # verify + export to Parquet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME
from db import init_db, get_collection


def verify_manifest():
    """Verify the master_manifest VIEW and underlying tables."""
    conn = init_db()
    
    print("📊 Table Row Counts:")
    tables = ["memories", "emotion_scores", "temporal_features", "place_assignments"]
    counts = {}
    for table in tables:
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        counts[table] = count
        print(f"   {table}: {count}")
    
    # Master manifest view
    manifest_count = conn.execute("SELECT count(*) FROM master_manifest").fetchone()[0]
    print(f"\n   master_manifest (VIEW): {manifest_count}")
    
    # Check consistency
    print("\n🔍 Integrity Checks:")
    
    # 1. Manifest should match memories count
    if manifest_count == counts["memories"]:
        print(f"   ✅ Manifest rows ({manifest_count}) = Memories rows ({counts['memories']})")
    else:
        print(f"   ❌ Manifest rows ({manifest_count}) ≠ Memories rows ({counts['memories']})")
    
    # 2. Check for NULLs in key columns
    null_checks = {
        "valence": "SELECT count(*) FROM master_manifest WHERE valence IS NULL",
        "arousal": "SELECT count(*) FROM master_manifest WHERE arousal IS NULL",
        "place_id": "SELECT count(*) FROM master_manifest WHERE place_id IS NULL",
        "sin_hour": "SELECT count(*) FROM master_manifest WHERE sin_hour IS NULL",
    }
    
    print("\n   NULL counts in master_manifest:")
    for col, query in null_checks.items():
        null_count = conn.execute(query).fetchone()[0]
        status = "⚠️" if null_count > 0 else "✅"
        print(f"   {status}  {col}: {null_count} NULLs")
    
    # 3. Check ChromaDB collections
    print("\n📦 ChromaDB Collections:")
    for coll_name in [IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME]:
        collection = get_collection(coll_name)
        coll_count = collection.count()
        print(f"   {coll_name}: {coll_count} vectors")
    
    # 4. Sample row
    print("\n📋 Sample Row (first record):")
    row = conn.execute("SELECT * FROM master_manifest LIMIT 1").fetchone()
    if row:
        columns = [desc[0] for desc in conn.execute("SELECT * FROM master_manifest LIMIT 0").description]
        for col in columns:
            val = row[columns.index(col)]
            if val is not None:
                print(f"   {col}: {val}")
    else:
        print("   (no records)")
    
    conn.close()
    return manifest_count > 0


def export_parquet():
    """Export master_manifest VIEW to a Parquet file for backward compatibility."""
    try:
        import pandas as pd
    except ImportError:
        print("❌ pandas is required for Parquet export: pip install pandas pyarrow")
        return
    
    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM master_manifest", conn)
    conn.close()
    
    output_path = PROCESSED_DIR / "master_manifest.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\n💾 Exported to: {output_path}")
    print(f"   Shape: {df.shape}")


def main():
    parser = argparse.ArgumentParser(description="Master Manifest Verification & Export")
    parser.add_argument("--export", action="store_true", help="Export VIEW to Parquet file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("DREAMS Research - Grand Fusion: Master Manifest")
    print("=" * 60)
    print()
    
    ok = verify_manifest()
    
    if args.export:
        print("\n📤 Exporting to Parquet...")
        export_parquet()
    
    print("\n" + "=" * 60)
    if ok:
        print("✅ Grand Fusion Verification Complete!")
    else:
        print("⚠️  No records found. Run the pipeline first.")
    print("=" * 60)


if __name__ == "__main__":
    main()
