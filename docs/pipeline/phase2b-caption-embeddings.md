# Phase 2B: Caption Embeddings

Extracts Sentence-BERT (MiniLM) text embeddings from memory captions.

## Run

```bash
source venv/bin/activate
pip install sentence-transformers>=2.2.0
python pipeline/extract_caption_embeddings.py
```

## Output

| File | Description |
|------|-------------|
| `data/processed/text_embeddings.npy` | (N, 384) Sentence-BERT embeddings |
| `data/processed/caption_embedding_index.json` | Record ID to embedding index mapping |

## Preprocessing Rules

- Unicode normalization (NFC)
- Strip leading/trailing whitespace
- Preserve punctuation and casing

## Model Details

- **Model**: all-MiniLM-L6-v2
- **Embedding Dimension**: 384
- **Normalization**: L2-normalized vectors
