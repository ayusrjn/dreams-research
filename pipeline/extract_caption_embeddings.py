import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, SENTENCE_BERT_MODEL, CAPTION_COLLECTION_NAME
from db import get_collection

def main():
    if not RAW_METADATA_PATH.exists():
        sys.exit(1)
    
    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)

    model = SentenceTransformer(SENTENCE_BERT_MODEL)

    captions = []
    record_infos = []

    for record in metadata.get("records", []):
        raw_caption = record.get("caption", "")
        if not raw_caption or raw_caption == "[REDACTED]":
            continue
        
        caption = unicodedata.normalize("NFC", raw_caption).strip()
        if not caption:
            continue
        
        captions.append(caption)
        record_infos.append({
            "id": str(record.get("id")),
            "user_id": record.get("user_id"),
            "caption": caption,
        })

    if not captions:
        return

    embeddings = model.encode(captions, convert_to_numpy=True, normalize_embeddings=True)

    collection = get_collection(CAPTION_COLLECTION_NAME)
    collection.upsert(
        ids=[r["id"] for r in record_infos],
        embeddings=embeddings.tolist(),
        documents=[r["caption"] for r in record_infos],
        metadatas=[{"user_id": r["user_id"]} for r in record_infos],
    )

if __name__ == "__main__":
    main()
