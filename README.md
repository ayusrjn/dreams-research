# DREAMS Research Pipeline

Memory research pipeline for disentangled feature extraction from captured memories.

## Project Structure

```
dreams-research/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   │   ├── user_01/          # Images organized by user
│   │   │   │   ├── img_001.jpg
│   │   │   │   └── img_002.jpg
│   │   │   └── user_02/
│   │   └── metadata.json         # All records with local image paths
│   │
│   ├── processed/                # Phase 2 outputs
│   │   ├── image_embeddings.npy
│   │   ├── text_embeddings.npy
│   │   ├── emotion_scores.csv
│   │   └── place_ids.csv
│   │
│   └── snapshots/                # Frozen experiment boundaries
│       └── snapshot_2026_01_25/
│
├── pipeline/                     # Processing scripts
│   ├── config.py
│   └── pull_data.py
│
├── analysis/                     # Analysis notebooks
└── README.md
```

## D1 Database Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Primary key |
| `user_id` | string | User UUID |
| `caption` | string | Memory caption |
| `timestamp` | datetime | When memory was captured |
| `lat` | float | Latitude |
| `lon` | float | Longitude |
| `image_url` | string | Cloudinary URL |
| `processed` | int | 0 = unprocessed |
| `processing_version` | string | Pipeline version |
| `created_at` | datetime | DB insert time |

## Phase 1: Data Pull & Freezing

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Cloudflare D1 credentials
```

### Run

```bash
source .env
python pipeline/pull_data.py
```

### Output

After running:
- `data/raw/images/{user_id}/` - Downloaded images per user
- `data/raw/metadata.json` - All records with local paths
- `data/snapshots/snapshot_YYYY_MM_DD/` - Frozen copy

The snapshot is the **experiment boundary** for Phase 2.

---

## Phases Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data Pull & Freezing | ✅ Ready |
| **Phase 2** | Feature Extraction (Image, Caption, Emotion, Time, Location) | 🔜 Planned |
