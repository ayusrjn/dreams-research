# DREAMS Research Pipeline

Memory research pipeline for disentangled feature extraction from captured memories.

## Project Structure

```text
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
│   ├── pull_data.py
│   └── extract_image_embeddings.py
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

## Phase 2A: Image Embeddings

Extracts CLIP (ViT-B/32) image embeddings from downloaded memories.

### Run

```bash
source venv/bin/activate
python pipeline/extract_image_embeddings.py
```

### Output

- `data/processed/image_embeddings.npy` - (N, 512) CLIP embeddings
- `data/processed/image_embedding_index.json` - Record ID to embedding index mapping

---

## Phase 2B: Caption Embeddings

Extracts Sentence-BERT (MiniLM) text embeddings from memory captions.

### Run

```bash
source venv/bin/activate
pip install sentence-transformers>=2.2.0
python pipeline/extract_caption_embeddings.py
```

### Output

- `data/processed/text_embeddings.npy` - (N, 384) Sentence-BERT embeddings
- `data/processed/caption_embedding_index.json` - Record ID to embedding index mapping

### Preprocessing Rules

- Unicode normalization (NFC)
- Strip leading/trailing whitespace
- Preserve punctuation and casing

---

## Phase 2C: Emotion Extraction

Extracts emotional features from captions using pretrained models:
- **Valence/Arousal**: Dimensional emotion (Mavdol/NPC-Valence-Arousal-Prediction)
- **Discrete Emotions**: Categorical probabilities (j-hartmann/emotion-english-distilroberta-base)

### Run

```bash
source venv/bin/activate
python pipeline/extract_emotions.py
```

### Output

- `data/processed/emotion_scores.csv` - Emotion features per record

### Columns

| Column | Description |
|--------|-------------|
| `id` | Unique record identifier |
| `user_id` | User identifier |
| `valence` | Pleasant (1) ↔ Unpleasant (0) |
| `arousal` | High energy (1) ↔ Low energy (0) |
| `joy` | Probability of joy |
| `sadness` | Probability of sadness |
| `fear` | Probability of fear |
| `anger` | Probability of anger |
| `neutral` | Probability of neutral |
| `disgust` | Probability of disgust |
| `surprise` | Probability of surprise |

> Emotion is an estimate of expressed affect, not internal state.

---

## Phase 2D: Temporal Representation

Extracts temporal features from timestamps for circadian and longitudinal analysis.

### Run

```bash
source venv/bin/activate
python pipeline/extract_temporal_features.py
```

### Output

- `data/processed/temporal_features.csv` - Temporal features per record

### Columns

| Column | Description |
|--------|-------------|
| `id` | Unique record identifier |
| `user_id` | User identifier |
| `absolute_utc` | ISO-8601 UTC timestamp |
| `relative_day` | Days since user's first entry |
| `sin_hour` | sin(2π × hour / 24) - Circadian X coordinate |
| `cos_hour` | cos(2π × hour / 24) - Circadian Y coordinate |

> Circadian encoding ensures 23:00 and 01:00 are mathematically close.

---

## Phases Overview

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data Pull & Freezing | ✅ Complete |
| **Phase 2A** | Image Embeddings (CLIP) | ✅ Complete |
| **Phase 2B** | Caption Embeddings (Sentence-BERT) | ✅ Complete |
| **Phase 2C** | Emotion Extraction | ✅ Complete |
| **Phase 2D** | Temporal Representation | ✅ Complete |
| **Phase 2E** | Location Clustering | 🔜 Planned |

