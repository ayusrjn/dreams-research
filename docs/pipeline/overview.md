# Pipeline Overview

The DREAMS pipeline consists of two main phases: data acquisition and feature extraction.

## Architecture

```mermaid
flowchart TD
    A[Phase 1: Data Pull] --> B[Raw Images + Metadata]
    B --> C[Snapshot Freeze]
    C --> D[Phase 2A: Image Embeddings]
    C --> E[Phase 2B: Caption Embeddings]
    C --> F[Phase 2C: Emotion Extraction]
    C --> G[Phase 2D: Temporal Features]
    C --> H[Phase 2E: Location Clustering]
    D --> I[processed/image_embeddings.npy]
    E --> J[processed/text_embeddings.npy]
    F --> K[processed/emotion_scores.csv]
    G --> L[processed/temporal_features.csv]
    H --> M[processed/place_ids.csv]
```

## Phase Summary

| Phase | Input | Output | Model |
|-------|-------|--------|-------|
| **1** | D1 Database | Raw images + metadata | - |
| **2A** | Images | 512-dim embeddings | CLIP ViT-B/32 |
| **2B** | Captions | 384-dim embeddings | Sentence-BERT |
| **2C** | Captions | Valence/arousal + emotions | DistilRoBERTa |
| **2D** | Timestamps | Circadian encoding | - |
| **2E** | GPS coords | Place IDs | DBSCAN |

## Data Flow

1. **Pull**: Download images and metadata from Cloudflare D1
2. **Freeze**: Create immutable snapshot for experiment reproducibility
3. **Extract**: Run feature extraction pipelines on frozen data
4. **Analyze**: Use extracted features for research analysis
