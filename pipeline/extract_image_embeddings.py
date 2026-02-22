#!/usr/bin/env python3
"""
Phase 2A: Image Embedding Extraction

Extracts CLIP image embeddings from downloaded memory images.
Uses the frozen snapshot from Phase 1 as input.
Stores embeddings in ChromaDB `image_embeddings` collection.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image
from PIL.Image import UnidentifiedImageError

# Import config from pipeline
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    RAW_IMAGES_DIR,
    CLIP_MODEL,
    IMAGE_COLLECTION_NAME,
)
from db import get_collection


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


def extract_embeddings(metadata: dict, model, preprocess, device) -> tuple[list[dict], np.ndarray]:
    """
    Extract image embeddings for all valid images.
    
    Returns:
        Tuple of (record_info_list, embeddings array)
    """
    records = metadata.get("records", [])
    embeddings = []
    record_infos = []
    
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
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image.load()
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                embedding = embedding.cpu().numpy().flatten()
            
            embeddings.append(embedding)
            record_infos.append({
                "id": str(record_id),
                "user_id": record.get("user_id"),
                "local_image": local_image,
            })
            
            print(f"   ✅ Record {record_id}: Extracted ({embedding.shape[0]} dims)")
            
        except (FileNotFoundError, UnidentifiedImageError, ValueError, RuntimeError) as e:
            print(f"   ❌ Record {record_id}: Failed - {e}")
            continue
    
    if not embeddings:
        return [], np.array([])
    
    return record_infos, np.stack(embeddings)


def store_embeddings(record_infos: list[dict], embeddings: np.ndarray) -> None:
    """Store embeddings in ChromaDB image_embeddings collection."""
    collection = get_collection(IMAGE_COLLECTION_NAME)
    
    ids = [r["id"] for r in record_infos]
    metadatas = [{"user_id": r["user_id"], "local_image": r["local_image"]} for r in record_infos]
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    
    print(f"💾 Stored {len(ids)} embeddings in ChromaDB ({IMAGE_COLLECTION_NAME})")
    print(f"   Collection size: {collection.count()}")


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
    record_infos, embeddings = extract_embeddings(metadata, model, preprocess, device)
    
    if len(embeddings) == 0:
        print("\n⚠️  No embeddings extracted. Check your images.")
        return
    
    # Step 4: Store in ChromaDB
    print("\n💾 Step 4: Storing in ChromaDB...")
    store_embeddings(record_infos, embeddings)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Phase 2A Complete!")
    print("=" * 60)
    print(f"   📊 Embeddings: {embeddings.shape[0]} images")
    print(f"   📐 Dimensions: {embeddings.shape[1]}")
    print(f"   💾 Collection: {IMAGE_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
