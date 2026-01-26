# DREAMS Research Pipeline

Memory research pipeline for disentangled feature extraction from captured memories.

## Overview

The DREAMS (Disentangled Representation Extraction for Autobiographical Memory Studies) pipeline processes multimodal memory data to extract rich feature representations for research analysis.

## Features

- **Image Embeddings** - CLIP ViT-B/32 visual representations
- **Caption Embeddings** - Sentence-BERT semantic representations  
- **Emotion Extraction** - Valence/arousal and discrete emotion scores
- **Temporal Features** - Circadian encoding for time-of-day analysis
- **Location Clustering** - DBSCAN-based place identification

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ayush-ranjan/dreams-research.git
cd dreams-research

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

## Pipeline Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data Pull & Freezing | ✅ Complete |
| **Phase 2A** | Image Embeddings (CLIP) | ✅ Complete |
| **Phase 2B** | Caption Embeddings (Sentence-BERT) | ✅ Complete |
| **Phase 2C** | Emotion Extraction | ✅ Complete |
| **Phase 2D** | Temporal Representation | ✅ Complete |
| **Phase 2E** | Location Clustering | ✅ Complete |
| **Phase 3** | Grand Fusion (Manifest + Vectors) | ✅ Complete |

## Project Structure

```text
dreams-research/
├── data/
│   ├── raw/              # Downloaded images and metadata
│   ├── processed/        # Extracted features
│   └── snapshots/        # Frozen experiment boundaries
├── pipeline/             # Processing scripts
├── analysis/             # Analysis notebooks
└── docs/                 # Documentation (you are here)
```
