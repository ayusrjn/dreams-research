# API Reference

## Pipeline Modules

### `pipeline/config.py`

Central configuration for all pipeline parameters.

### `pipeline/pull_data.py`

Data acquisition from Cloudflare D1 database.

### `pipeline/extract_image_embeddings.py`

CLIP image embedding extraction.

### `pipeline/extract_caption_embeddings.py`

Sentence-BERT text embedding extraction.

### `pipeline/extract_emotions.py`

Emotion score extraction using transformer models.

### `pipeline/extract_temporal_features.py`

Temporal feature engineering.

## Output Formats

### NumPy Arrays (`.npy`)

Used for high-dimensional embeddings:

```python
import numpy as np
embeddings = np.load('data/processed/image_embeddings.npy')
# Shape: (N, embedding_dim)
```

### CSV Files

Used for tabular features:

```python
import pandas as pd
emotions = pd.read_csv('data/processed/emotion_scores.csv')
```

### JSON Index Files

Maps record IDs to array indices:

```python
import json
with open('data/processed/image_embedding_index.json') as f:
    index = json.load(f)
# {"record_id": array_index, ...}
```
