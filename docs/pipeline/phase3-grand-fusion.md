# Phase 3: Grand Fusion

The **Grand Fusion** is the final step of the DREAMS Research Pipeline. It merges all extracted features into a single **Master Manifest** and synchronizes high-dimensional vectors to match the sorted manifest.

## Overview

This phase performs three critical actions:

1.  **Relational Join**: Merges metadata, emotion scores, temporal features, and place IDs into a single dataframe.
2.  **Vector Alignment**: Re-orders image and text vectors to match the sorted order of the master manifest.
3.  **Normalization**: Ensures all features are in the correct format (e.g., categorical place IDs).

## Input Files

| File | Description | Source |
|------|-------------|--------|
| `data/raw/metadata.json` | Original memory records | Phase 1 |
| `data/processed/emotion_scores.csv` | Valence, arousal, and discrete emotions | Phase 2C |
| `data/processed/temporal_features.csv` | Circadian and longitudinal features | Phase 2D |
| `data/processed/place_ids.csv` | Clustered location IDs | Phase 2E |
| `data/processed/image_embeddings.npy` | CLIP image vectors (N, 512) | Phase 2A |
| `data/processed/text_embeddings.npy` | S-BERT text vectors (N, 384) | Phase 2B |

## Execution

To run the Grand Fusion:

```bash
source venv/bin/activate
pip install pandas pyarrow
python pipeline/create_master_manifest.py
```

## Outputs

### 1. Master Manifest

**Path**: `data/processed/master_manifest.parquet`

A Parquet file containing the unified dataset. Key columns include:

-   `id`: Unique record identifier (Primary Key)
-   `timestamp`: ISO-8601 UTC timestamp
-   `valence`, `arousal`: Emotional dimensions (0-1)
-   `joy`, `sadness`, etc.: Discrete emotion probabilities
-   `sin_hour`, `cos_hour`: Circadian time encoding
-   `place_id`: Categorical location cluster ID

### 2. Synchronized Vectors

**Paths**:
-   `data/processed/final_image_vectors.npy`
-   `data/processed/final_text_vectors.npy`

These are numpy arrays where the vector at index `i` corresponds exactly to the record at row `i` in the Master Manifest.

## Verification

To verify that the manifest and vectors are perfectly aligned:

```bash
python pipeline/verify_alignment.py
```

This script checks:
1.  **Length Consistency**: `len(df) == len(img_vecs) == len(txt_vecs)`
2.  **Semantic Sanity**: Performs a cosine similarity check on the first record to ensure the text vector matches its caption.
