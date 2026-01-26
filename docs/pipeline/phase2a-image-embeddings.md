# Phase 2A: Image Embeddings

Extracts CLIP (ViT-B/32) image embeddings from downloaded memories.

## Run

```bash
source venv/bin/activate
python pipeline/extract_image_embeddings.py
```

## Output

| File | Description |
|------|-------------|
| `data/processed/image_embeddings.npy` | (N, 512) CLIP embeddings |
| `data/processed/image_embedding_index.json` | Record ID to embedding index mapping |

## Model Details

- **Model**: OpenAI CLIP ViT-B/32
- **Embedding Dimension**: 512
- **Normalization**: L2-normalized vectors

## Usage

```python
import numpy as np
import json

# Load embeddings
embeddings = np.load('data/processed/image_embeddings.npy')
with open('data/processed/image_embedding_index.json') as f:
    index = json.load(f)

# Get embedding for a specific record
record_id = "123"
idx = index[record_id]
embedding = embeddings[idx]
```
