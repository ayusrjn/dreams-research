#!/usr/bin/env python3
"""
Phase 2E: Location Clustering

Transforms raw GPS coordinates into categorical Place IDs using DBSCAN clustering.
Uses the frozen snapshot from Phase 1 as input.

Preprocessing:
    - Snap-to-grid: Truncate lat/lon to 4 decimal places (~11m buffer)

Clustering:
    - Algorithm: DBSCAN with haversine metric
    - Epsilon: 0.0005 (~50m radius)
    - Min Samples: 1 (single entries are valid data points)

Output:
    data/processed/
        place_ids.csv - Place ID assignments per record
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    PROCESSED_DIR,
    PLACE_IDS_PATH,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    COORD_DECIMAL_PLACES,
)


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
    
    Args:
        value: Raw coordinate value
        decimals: Number of decimal places to keep
        
    Returns:
        Truncated coordinate value
    """
    multiplier = 10 ** decimals
    return int(value * multiplier) / multiplier


def extract_coordinates(metadata: dict) -> tuple[list[dict], np.ndarray]:
    """
    Extract and preprocess coordinates from metadata.
    
    Returns:
        Tuple of (records with valid coords, coordinate array for clustering)
    """
    records = metadata.get("records", [])
    valid_records = []
    coordinates = []
    
    print(f"📍 Processing {len(records)} records...")
    
    for record in records:
        record_id = record.get("id")
        lat = record.get("lat")
        lon = record.get("lon")
        
        # Skip records without valid coordinates
        if lat is None or lon is None:
            print(f"   ⚠️  Record {record_id}: No coordinates (skipped)")
            continue
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            print(f"   ⚠️  Record {record_id}: Invalid coordinates (skipped)")
            continue
        
        # Snap to grid for noise reduction
        snapped_lat = snap_to_grid(lat, COORD_DECIMAL_PLACES)
        snapped_lon = snap_to_grid(lon, COORD_DECIMAL_PLACES)
        
        valid_records.append({
            "id": record_id,
            "user_id": record.get("user_id"),
            "raw_lat": lat,
            "raw_lon": lon,
            "snapped_lat": snapped_lat,
            "snapped_lon": snapped_lon,
        })
        
        # Convert to radians for haversine metric
        coordinates.append([np.radians(snapped_lat), np.radians(snapped_lon)])
    
    return valid_records, np.array(coordinates)


def cluster_locations(coordinates: np.ndarray) -> np.ndarray:
    """
    Cluster coordinates using DBSCAN with haversine metric.
    
    Args:
        coordinates: Array of (lat, lon) in radians
        
    Returns:
        Array of cluster labels (-1 for noise points)
    """
    if len(coordinates) == 0:
        return np.array([])
    
    print(f"🔍 Clustering {len(coordinates)} coordinates...")
    print(f"   Epsilon: {DBSCAN_EPS} (~50m)")
    print(f"   Min samples: {DBSCAN_MIN_SAMPLES}")
    
    # DBSCAN with haversine metric (requires radians input)
    # Note: sklearn's haversine expects (lat, lon) in radians
    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree"
    )
    
    labels = clustering.fit_predict(coordinates)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"   Found {n_clusters} clusters, {n_noise} noise points")
    
    return labels


def compute_centroids(records: list[dict], labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """
    Compute centroid (mean lat/lon) for each cluster.
    
    Args:
        records: List of record dictionaries with coordinates
        labels: Cluster labels array
        
    Returns:
        Dictionary mapping cluster label to (centroid_lat, centroid_lon)
    """
    cluster_points = {}
    
    for record, label in zip(records, labels):
        if label not in cluster_points:
            cluster_points[label] = []
        cluster_points[label].append((record["raw_lat"], record["raw_lon"]))
    
    centroids = {}
    for label, points in cluster_points.items():
        mean_lat = sum(p[0] for p in points) / len(points)
        mean_lon = sum(p[1] for p in points) / len(points)
        centroids[label] = (round(mean_lat, 6), round(mean_lon, 6))
    
    return centroids


def assign_place_ids(records: list[dict], labels: np.ndarray, centroids: dict) -> list[dict]:
    """
    Assign stable place IDs to records.
    
    Args:
        records: List of record dictionaries
        labels: Cluster labels array
        centroids: Dictionary of cluster centroids
        
    Returns:
        List of result dictionaries with place ID assignments
    """
    results = []
    
    # Create stable place_id mapping (sorted by first occurrence)
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    
    # Noise points (-1) get individual place IDs
    noise_counter = 0
    label_to_place_id = {}
    place_counter = 1
    
    for label in unique_labels:
        if label == -1:
            continue  # Handle noise points individually
        label_to_place_id[label] = f"place_{place_counter:02d}"
        place_counter += 1
    
    for record, label in zip(records, labels):
        if label == -1:
            # Each noise point gets its own place ID
            place_id = f"place_{place_counter + noise_counter:02d}"
            noise_counter += 1
            centroid = (record["raw_lat"], record["raw_lon"])
        else:
            place_id = label_to_place_id[label]
            centroid = centroids[label]
        
        result = {
            "id": record["id"],
            "user_id": record["user_id"],
            "raw_lat": round(record["raw_lat"], 6),
            "raw_lon": round(record["raw_lon"], 6),
            "place_id": place_id,
            "centroid_lat": centroid[0],
            "centroid_lon": centroid[1],
            "is_new_cluster": False,  # For future incremental processing
        }
        results.append(result)
        
        print(f"   ✅ Record {record['id']}: {place_id} (centroid: {centroid[0]:.4f}, {centroid[1]:.4f})")
    
    return results


def save_outputs(results: list[dict]) -> None:
    """Save place ID assignments to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = ["id", "user_id", "raw_lat", "raw_lon", "place_id", "centroid_lat", "centroid_lon", "is_new_cluster"]
    
    with open(PLACE_IDS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow([r.get(h, "") for h in headers])
    
    print(f"💾 Place IDs saved: {PLACE_IDS_PATH}")
    print(f"   Records: {len(results)}")


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
    
    # Step 2: Extract and preprocess coordinates
    print("\n📍 Step 2: Extracting coordinates...")
    records, coordinates = extract_coordinates(metadata)
    
    if len(records) == 0:
        print("\n⚠️  No valid coordinates found. Check your data.")
        return
    
    print(f"   Valid records: {len(records)}")
    
    # Step 3: Cluster locations
    print("\n🔍 Step 3: Clustering locations...")
    labels = cluster_locations(coordinates)
    
    # Step 4: Compute centroids
    print("\n📐 Step 4: Computing centroids...")
    centroids = compute_centroids(records, labels)
    
    # Step 5: Assign place IDs
    print("\n🏷️  Step 5: Assigning place IDs...")
    results = assign_place_ids(records, labels, centroids)
    
    # Step 6: Save outputs
    print("\n💾 Step 6: Saving outputs...")
    save_outputs(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 2E Complete!")
    print("=" * 60)
    
    unique_places = len(set(r["place_id"] for r in results))
    print(f"   📊 Processed: {len(results)} records")
    print(f"   🏘️  Unique places: {unique_places}")
    print(f"   📁 Output: {PLACE_IDS_PATH}")


if __name__ == "__main__":
    main()
