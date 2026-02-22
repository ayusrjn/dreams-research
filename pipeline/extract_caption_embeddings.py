#!/usr/bin/env python3
"""
Phase 2B: Caption Embedding Extraction

Extracts Sentence-BERT embeddings from memory captions.
Uses the frozen snapshot from Phase 1 as input.
Stores embeddings in ChromaDB `caption_embeddings` collection.
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Import config from pipeline
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    SENTENCE_BERT_MODEL,
    CAPTION_COLLECTION_NAME,
)
from db import get_collection


def load_metadata() -> dict:
    """Load the frozen snapshot metadata."""
    if not RAW_METADATA_PATH.exists():
        print(f"[ERROR] Metadata not found: {RAW_METADATA_PATH}")
        print("   Run Phase 1 first: python pipeline/pull_data.py")
        sys.exit(1)
    
    with open(RAW_METADATA_PATH) as f:
        return json.load(f)


def preprocess_caption(text: str) -> str:
    """
    Preprocess caption text for embedding extraction.
    
    Rules (per plan.md):
        - Strip leading/trailing whitespace
        - Normalize Unicode (NFC)
        - Keep punctuation
        - Keep casing (MiniLM is case-sensitive)
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    return text


def load_sentence_bert() -> SentenceTransformer:
    """Load Sentence-BERT model."""
    print(f"[INFO] Loading Sentence-BERT model: {SENTENCE_BERT_MODEL}")
    model = SentenceTransformer(SENTENCE_BERT_MODEL)
    return model


def extract_embeddings(metadata: dict, model: SentenceTransformer) -> tuple[list[dict], np.ndarray]:
    """
    Extract caption embeddings for all records with valid captions.
    
    Returns:
        Tuple of (record_info_list, embeddings array)
    """
    records = metadata.get("records", [])
    captions = []
    record_infos = []
    
 print(f" Processing {len(records)} records...")
    
    for record in records:
        record_id = record.get("id")
        raw_caption = record.get("caption", "")
        
        if not raw_caption or raw_caption == "[REDACTED]":
            print(f"   [WARN] Record {record_id}: No valid caption (skipped)")
            continue
        
        caption = preprocess_caption(raw_caption)
        
        if not caption:
            print(f"   [WARN] Record {record_id}: Empty after preprocessing (skipped)")
            continue
        
        captions.append(caption)
        record_infos.append({
            "id": str(record_id),
            "user_id": record.get("user_id"),
            "caption": caption,
        })
        print(f"   [OK] Record {record_id}: '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
    
    if not captions:
        return [], np.array([])
    
    # Batch encode all captions
    print(f"\n[INFO] Encoding {len(captions)} captions...")
    embeddings = model.encode(
        captions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # Unit-normalized vectors per plan.md
    )
    
    return record_infos, embeddings


def store_embeddings(record_infos: list[dict], embeddings: np.ndarray) -> None:
    """Store embeddings in ChromaDB caption_embeddings collection."""
    collection = get_collection(CAPTION_COLLECTION_NAME)
    
    ids = [r["id"] for r in record_infos]
    documents = [r["caption"] for r in record_infos]
    metadatas = [{"user_id": r["user_id"]} for r in record_infos]
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )
    
    print(f"[INFO] Stored {len(ids)} embeddings in ChromaDB ({CAPTION_COLLECTION_NAME})")
    print(f"   Collection size: {collection.count()}")


def main():
    """Main execution flow for Phase 2B: Caption Embeddings."""
    print("=" * 60)
    print("DREAMS Research - Phase 2B: Caption Embedding Extraction")
    print("=" * 60)
    print()
    
    # Step 1: Load metadata
 print("[INFO] Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")
    
    # Step 2: Load Sentence-BERT model
    print("\n[INFO] Step 2: Loading Sentence-BERT model...")
    model = load_sentence_bert()
    
    # Step 3: Extract embeddings
    print("\n[INFO] Step 3: Extracting embeddings...")
    record_infos, embeddings = extract_embeddings(metadata, model)
    
    if len(embeddings) == 0:
        print("\n[WARN] No embeddings extracted. Check your captions.")
        print("   Note: [REDACTED] captions are skipped.")
        return
    
    # Step 4: Store in ChromaDB
    print("\n[INFO] Step 4: Storing in ChromaDB...")
    store_embeddings(record_infos, embeddings)
    
    # Summary
    print("\n" + "=" * 60)
    print("[OK] Phase 2B Complete!")
    print("=" * 60)
    print(f"   [INFO] Embeddings: {embeddings.shape[0]} captions")
    print(f"   [INFO] Dimensions: {embeddings.shape[1]}")
    print(f"   [INFO] Collection: {CAPTION_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
