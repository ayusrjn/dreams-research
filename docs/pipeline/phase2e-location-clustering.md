# Phase 2E: Location Clustering

Clusters raw GPS coordinates into categorical Place IDs using DBSCAN.

## Run

```bash
source venv/bin/activate
python pipeline/extract_location_clusters.py
```

## Output

`data/processed/place_ids.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique record identifier |
| `user_id` | User identifier |
| `raw_lat` | Original latitude |
| `raw_lon` | Original longitude |
| `place_id` | Cluster identifier (e.g., `place_01`) |
| `centroid_lat` | Cluster centroid latitude |
| `centroid_lon` | Cluster centroid longitude |
| `is_new_cluster` | For incremental processing |

## Algorithm

1. **Snap-to-grid**: Truncate coordinates to 4 decimal places (~11m buffer)
2. **DBSCAN**: Haversine metric, ε ≈ 7.85×10⁻⁶ radians (~50m), min_samples=1

!!! info "Design Choice"
    Location is treated as **categorical context**, not a continuous vector. This prevents overfitting to GPS noise.

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid precision | 4 decimal places | ~11m spatial buffer |
| DBSCAN ε | 7.85×10⁻⁶ rad | ~50m radius |
| min_samples | 1 | Single-point clusters allowed |
