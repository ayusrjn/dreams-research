#!/usr/bin/env python3
"""
Phase 2A: Image Embedding Extraction

Extracts CLIP image embeddings from downloaded memory images.
Uses the frozen snapshot from Phase 1 as input.

Output:
    data/processed/
        image_embeddings.npy      - (N, 512) array of embeddings
        image_embedding_index.json - Record ID to embedding index mapping
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

# Import config from pipeline
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    RAW_IMAGES_DIR,
    PROCESSED_DIR,
    IMAGE_EMBEDDINGS_PATH,
    CLIP_MODEL,
)

# Output paths
EMBEDDING_INDEX_PATH = PROCESSED_DIR / "image_embedding_index.json"


def load_metadata() -> dict:
    """Load the frozen snapshot metadata."""
    if not RAW_METADATA_PATH.exists():
        print(f"❌ Metadata not found: {RAW_METADATA_PATH}")
        print("   Run Phase 1 first: python pipeline/pull_data.py")
        sys.exit(1)
    
    with open(RAW_METADATA_PATH) as f:
        return json.load(f)


def load_clip_model():
    """Load CLIP model and preprocessing function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading CLIP model: {CLIP_MODEL} on {device}")
    
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()
    
    return model, preprocess, device


def extract_embeddings(metadata: dict, model, preprocess, device) -> tuple[np.ndarray, dict]:
    """
    Extract image embeddings for all valid images.
    
    Returns:
        Tuple of (embeddings array, index mapping)
    """
    records = metadata.get("records", [])
    embeddings = []
    index_mapping = {}
    
    print(f"📷 Processing {len(records)} records...")
    
    for record in records:
        record_id = record.get("id")
        local_image = record.get("local_image")
        
        if not local_image:
            print(f"   ⚠️  Record {record_id}: No local image path")
            continue
        
        # Try exact path first
        image_path = RAW_IMAGES_DIR / local_image
        
        # If not found, search by filename in all subdirectories
        if not image_path.exists():
            filename = Path(local_image).name
            found_paths = list(RAW_IMAGES_DIR.rglob(filename))
            if found_paths:
                image_path = found_paths[0]
            else:
                print(f"   ⚠️  Record {record_id}: Image not found - {local_image}")
                continue
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embedding = embedding.cpu().numpy().flatten()
            
            # Store embedding and update index
            embedding_idx = len(embeddings)
            embeddings.append(embedding)
            index_mapping[str(record_id)] = {
                "embedding_index": embedding_idx,
                "user_id": record.get("user_id"),
                "local_image": local_image,
            }
            
            print(f"   ✅ Record {record_id}: Extracted ({embedding.shape[0]} dims)")
            
        except Exception as e:
            print(f"   ❌ Record {record_id}: Failed - {e}")
            continue
    
    if not embeddings:
        return np.array([]), {}
    
    return np.stack(embeddings), index_mapping


def save_outputs(embeddings: np.ndarray, index_mapping: dict) -> None:
    """Save embeddings and index mapping."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(IMAGE_EMBEDDINGS_PATH, embeddings)
    print(f"💾 Embeddings saved: {IMAGE_EMBEDDINGS_PATH}")
    print(f"   Shape: {embeddings.shape}")
    
    # Save index mapping
    with open(EMBEDDING_INDEX_PATH, "w") as f:
        json.dump(index_mapping, f, indent=2)
    print(f"💾 Index saved: {EMBEDDING_INDEX_PATH}")


def main():
    """Main execution flow for Phase 2A: Image Embeddings."""
    print("=" * 60)
    print("DREAMS Research - Phase 2A: Image Embedding Extraction")
    print("=" * 60)
    print()
    
    # Step 1: Load metadata
    print("📂 Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")
    
    # Step 2: Load CLIP model
    print("\n🧠 Step 2: Loading CLIP model...")
    model, preprocess, device = load_clip_model()
    
    # Step 3: Extract embeddings
    print("\n🔍 Step 3: Extracting embeddings...")
    embeddings, index_mapping = extract_embeddings(metadata, model, preprocess, device)
    
    if len(embeddings) == 0:
        print("\n⚠️  No embeddings extracted. Check your images.")
        return
    
    # Step 4: Save outputs
    print("\n💾 Step 4: Saving outputs...")
    save_outputs(embeddings, index_mapping)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 2A Complete!")
    print("=" * 60)
    print(f"   📊 Embeddings: {embeddings.shape[0]} images")
    print(f"   📐 Dimensions: {embeddings.shape[1]}")
    print(f"   📁 Output: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
