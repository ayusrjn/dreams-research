#!/usr/bin/env python3
"""
Phase 2B: Caption Embedding Extraction

Extracts Sentence-BERT embeddings from memory captions.
Uses the frozen snapshot from Phase 1 as input.

Output:
    data/processed/
        text_embeddings.npy           - (N, 384) array of embeddings
        caption_embedding_index.json  - Record ID to embedding index mapping
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
    PROCESSED_DIR,
    TEXT_EMBEDDINGS_PATH,
    CAPTION_EMBEDDING_INDEX_PATH,
    SENTENCE_BERT_MODEL,
)


def load_metadata() -> dict:
    """Load the frozen snapshot metadata."""
    if not RAW_METADATA_PATH.exists():
        print(f"❌ Metadata not found: {RAW_METADATA_PATH}")
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
    
    Args:
        text: Raw caption text
        
    Returns:
        Preprocessed caption string
    """
    if not text:
        return ""
    
    # Normalize Unicode to NFC form
    text = unicodedata.normalize("NFC", text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def load_sentence_bert() -> SentenceTransformer:
    """Load Sentence-BERT model."""
    print(f"🔧 Loading Sentence-BERT model: {SENTENCE_BERT_MODEL}")
    model = SentenceTransformer(SENTENCE_BERT_MODEL)
    return model


def extract_embeddings(metadata: dict, model: SentenceTransformer) -> tuple[np.ndarray, dict]:
    """
    Extract caption embeddings for all records with valid captions.
    
    Returns:
        Tuple of (embeddings array, index mapping)
    """
    records = metadata.get("records", [])
    captions = []
    record_ids = []
    index_mapping = {}
    
    print(f"📝 Processing {len(records)} records...")
    
    # Collect and preprocess captions
    for record in records:
        record_id = record.get("id")
        raw_caption = record.get("caption", "")
        
        # Skip redacted or empty captions
        if not raw_caption or raw_caption == "[REDACTED]":
            print(f"   ⚠️  Record {record_id}: No valid caption (skipped)")
            continue
        
        caption = preprocess_caption(raw_caption)
        
        if not caption:
            print(f"   ⚠️  Record {record_id}: Empty after preprocessing (skipped)")
            continue
        
        captions.append(caption)
        record_ids.append(record)
        print(f"   ✅ Record {record_id}: '{caption[:50]}{'...' if len(caption) > 50 else ''}'")
    
    if not captions:
        return np.array([]), {}
    
    # Batch encode all captions
    print(f"\n🧠 Encoding {len(captions)} captions...")
    embeddings = model.encode(
        captions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # Unit-normalized vectors per plan.md
    )
    
    # Build index mapping
    for idx, record in enumerate(record_ids):
        record_id = str(record.get("id"))
        index_mapping[record_id] = {
            "embedding_index": idx,
            "user_id": record.get("user_id"),
            "caption_preview": captions[idx][:100],
        }
    
    return embeddings, index_mapping


def save_outputs(embeddings: np.ndarray, index_mapping: dict) -> None:
    """Save embeddings and index mapping."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(TEXT_EMBEDDINGS_PATH, embeddings)
    print(f"💾 Embeddings saved: {TEXT_EMBEDDINGS_PATH}")
    print(f"   Shape: {embeddings.shape}")
    
    # Save index mapping
    with open(CAPTION_EMBEDDING_INDEX_PATH, "w") as f:
        json.dump(index_mapping, f, indent=2)
    print(f"💾 Index saved: {CAPTION_EMBEDDING_INDEX_PATH}")


def main():
    """Main execution flow for Phase 2B: Caption Embeddings."""
    print("=" * 60)
    print("DREAMS Research - Phase 2B: Caption Embedding Extraction")
    print("=" * 60)
    print()
    
    # Step 1: Load metadata
    print("📂 Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")
    
    # Step 2: Load Sentence-BERT model
    print("\n🧠 Step 2: Loading Sentence-BERT model...")
    model = load_sentence_bert()
    
    # Step 3: Extract embeddings
    print("\n🔍 Step 3: Extracting embeddings...")
    embeddings, index_mapping = extract_embeddings(metadata, model)
    
    if len(embeddings) == 0:
        print("\n⚠️  No embeddings extracted. Check your captions.")
        print("   Note: [REDACTED] captions are skipped.")
        return
    
    # Step 4: Save outputs
    print("\n💾 Step 4: Saving outputs...")
    save_outputs(embeddings, index_mapping)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 2B Complete!")
    print("=" * 60)
    print(f"   📊 Embeddings: {embeddings.shape[0]} captions")
    print(f"   📐 Dimensions: {embeddings.shape[1]}")
    print(f"   📁 Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
