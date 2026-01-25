"""
Configuration constants for DREAMS Research pipeline.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Phase 1: Raw data paths
RAW_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DIR / "images"
RAW_METADATA_PATH = RAW_DIR / "metadata.json"

# Phase 2: Processed data paths
PROCESSED_DIR = DATA_DIR / "processed"
IMAGE_EMBEDDINGS_PATH = PROCESSED_DIR / "image_embeddings.npy"
TEXT_EMBEDDINGS_PATH = PROCESSED_DIR / "text_embeddings.npy"
EMOTION_SCORES_PATH = PROCESSED_DIR / "emotion_scores.csv"
PLACE_IDS_PATH = PROCESSED_DIR / "place_ids.csv"

# Snapshots
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Model configurations (Phase 2)
CLIP_MODEL = "ViT-B/32"
SENTENCE_BERT_MODEL = "all-MiniLM-L6-v2"

# D1 Database Schema
D1_SCHEMA = {
    "table": "memories",
    "columns": [
        "id",
        "user_id",
        "caption",
        "timestamp",
        "lat",
        "lon",
        "image_url",
        "processed",
        "processing_version",
        "created_at"
    ]
}

# Emotion representation schema
EMOTION_SCHEMA = {
    "valence": float,
    "arousal": float,
    "confidence": float
}
