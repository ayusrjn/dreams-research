#!/usr/bin/env python3
"""
Phase 2E: Location Clustering

Transforms raw GPS coordinates into categorical Place IDs using DBSCAN clustering.
Uses the frozen snapshot from Phase 1 as input.

Preprocessing:
    - Snap-to-grid: Truncate coordinates to 4 decimal places (~11m precision)
    
Algorithm:
    - DBSCAN with haversine metric
    - Epsilon ≈ 50m
    - min_samples = 1 (single entries are valid data points)

Stores results in the SQLite `place_assignments` table.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    COORD_DECIMAL_PLACES,
)
from db import init_db


def load_metadata() -> dict:
    """Load the frozen snapshot metadata."""
    if not RAW_METADATA_PATH.exists():
        print(f"❌ Metadata not found: {RAW_METADATA_PATH}")
        print("   Run Phase 1 first: python pipeline/pull_data.py")
        sys.exit(1)
    
    with open(RAW_METADATA_PATH) as f:
        return json.load(f)


def snap_to_grid(value: float, decimals: int = 4) -> float:
    """
    Truncate coordinate to specified decimal places.
    
    This provides an ~11m buffer against GPS sensor noise when using 4 decimals.
    """
    factor = 10 ** decimals
    return math.trunc(value * factor) / factor


def extract_coordinates(metadata: dict) -> tuple[list[dict], np.ndarray]:
    """
    Extract and preprocess coordinates from metadata.
    
    Returns:
        Tuple of (records with valid coords, coordinate array for clustering)
    """
    records = metadata.get("records", [])
    valid_records = []
    coords = []
    
    print(f"📍 Processing {len(records)} records...")
    
    for record in records:
        record_id = record.get("id")
        user_id = record.get("user_id")
        lat = record.get("lat")
        lon = record.get("lon")
        
        if lat is None or lon is None:
            print(f"   ⚠️  Record {record_id}: No coordinates (skipped)")
            continue
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            print(f"   ⚠️  Record {record_id}: Invalid coordinates (skipped)")
            continue
        
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            print(f"   ⚠️  Record {record_id}: Out of range ({lat}, {lon}) (skipped)")
            continue
        
        snapped_lat = snap_to_grid(lat, COORD_DECIMAL_PLACES)
        snapped_lon = snap_to_grid(lon, COORD_DECIMAL_PLACES)
        
        valid_records.append({
            "id": record_id,
            "user_id": user_id,
            "raw_lat": lat,
            "raw_lon": lon,
            "snapped_lat": snapped_lat,
            "snapped_lon": snapped_lon,
        })
        
        coords.append([math.radians(snapped_lat), math.radians(snapped_lon)])
    
    print(f"   Valid records: {len(valid_records)}/{len(records)}")
    
    if not coords:
        return [], np.array([])
    
    return valid_records, np.array(coords)


def cluster_locations(coordinates: np.ndarray) -> np.ndarray:
    """
    Cluster coordinates using DBSCAN with haversine metric.
    
    Returns:
        Array of cluster labels (-1 for noise points)
    """
    print(f"   DBSCAN parameters:")
    print(f"      eps = {DBSCAN_EPS:.10f} radians (~50m)")
    print(f"      min_samples = {DBSCAN_MIN_SAMPLES}")
    
    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    
    labels = clustering.fit_predict(coordinates)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    
    print(f"   Clusters found: {n_clusters}")
    if n_noise > 0:
        print(f"   Noise points: {n_noise}")
    
    return labels


def compute_centroids(records: list[dict], labels: np.ndarray) -> dict:
    """Compute centroid (mean lat/lon) for each cluster."""
    cluster_points = {}
    
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in cluster_points:
            cluster_points[label] = []
        cluster_points[label].append((records[i]["snapped_lat"], records[i]["snapped_lon"]))
    
    centroids = {}
    for label, points in cluster_points.items():
        lats, lons = zip(*points)
        centroids[label] = (sum(lats) / len(lats), sum(lons) / len(lons))
    
    return centroids


def assign_place_ids(records: list[dict], labels: np.ndarray, centroids: dict) -> list[dict]:
    """Assign stable place IDs to records."""
    results = []
    place_counter = 0
    label_to_place = {}
    
    for i, (record, label) in enumerate(zip(records, labels)):
        if label == -1:
            place_id = f"place_solo_{record['id']}"
            centroid_lat = record["snapped_lat"]
            centroid_lon = record["snapped_lon"]
            is_new = 1
        else:
            if label not in label_to_place:
                place_counter += 1
                label_to_place[label] = f"place_{place_counter:03d}"
            
            place_id = label_to_place[label]
            centroid_lat, centroid_lon = centroids[label]
            is_new = 0
        
        results.append({
            "id": record["id"],
            "user_id": record["user_id"],
            "snapped_lat": record["snapped_lat"],
            "snapped_lon": record["snapped_lon"],
            "place_id": place_id,
            "centroid_lat": round(centroid_lat, 6),
            "centroid_lon": round(centroid_lon, 6),
            "is_new_cluster": is_new,
        })
        
        print(f"   ✅ Record {record['id']}: {place_id} "
              f"({record['snapped_lat']:.4f}, {record['snapped_lon']:.4f})")
    
    return results


def store_place_assignments(results: list[dict]) -> None:
    """Store place assignments in the SQLite place_assignments table."""
    conn = init_db()
    
    for r in results:
        conn.execute(
            """INSERT OR REPLACE INTO place_assignments
               (id, user_id, snapped_lat, snapped_lon,
                place_id, centroid_lat, centroid_lon, is_new_cluster)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["id"], r["user_id"], r["snapped_lat"], r["snapped_lon"],
                r["place_id"], r["centroid_lat"], r["centroid_lon"],
                r["is_new_cluster"],
            ),
        )
    
    conn.commit()
    conn.close()
    
    print(f"💾 Stored {len(results)} place assignments in SQLite (place_assignments)")


def main():
    """Main execution flow for Phase 2E: Location Clustering."""
    print("=" * 60)
    print("DREAMS Research - Phase 2E: Location Clustering")
    print("=" * 60)
    print()
    
    # Step 1: Load metadata
    print("📂 Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")
    
    # Step 2: Extract coordinates
    print("\n📍 Step 2: Extracting and preprocessing coordinates...")
    records, coordinates = extract_coordinates(metadata)
    
    if len(records) == 0:
        print("\n⚠️  No valid coordinates found.")
        return
    
    # Step 3: Cluster
    print("\n🔗 Step 3: Clustering locations...")
    labels = cluster_locations(coordinates)
    
    # Step 4: Compute centroids
    print("\n📌 Step 4: Computing centroids...")
    centroids = compute_centroids(records, labels)
    print(f"   Centroids computed for {len(centroids)} clusters")
    
    # Step 5: Assign Place IDs
    print("\n🏷️  Step 5: Assigning Place IDs...")
    results = assign_place_ids(records, labels, centroids)
    
    # Step 6: Store in SQLite
    print("\n💾 Step 6: Storing in SQLite...")
    store_place_assignments(results)
    
    # Summary
    unique_places = len(set(r["place_id"] for r in results))
    print("\n" + "=" * 60)
    print("✅ Phase 2E Complete!")
    print("=" * 60)
    print(f"   📊 Records: {len(results)}")
    print(f"   🗺️  Unique places: {unique_places}")


if __name__ == "__main__":
    main()
