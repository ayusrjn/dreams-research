# Phase 1: Data Pull & Freezing

Pull memory data from the cloud database and create frozen snapshots for reproducible experiments.

## Run

```bash
source venv/bin/activate
source .env
python pipeline/pull_data.py
```

## Output

After running:

- `data/raw/images/{user_id}/` - Downloaded images per user
- `data/raw/metadata.json` - All records with local paths
- `data/snapshots/snapshot_YYYY_MM_DD/` - Frozen copy

!!! tip "Experiment Boundary"
    The snapshot is the **experiment boundary** for Phase 2. All feature extraction operates on this frozen data.

## Data Structure

```text
data/
├── raw/
│   ├── images/
│   │   ├── user_01/
│   │   │   ├── img_001.jpg
│   │   │   └── img_002.jpg
│   │   └── user_02/
│   └── metadata.json
└── snapshots/
    └── snapshot_2026_01_25/
```
