#!/usr/bin/env python3
"""
Grand Fusion: Create Master Manifest & Synchronize Vectors

Merges all feature files into a single Master Manifest and aligns
high-dimensional vectors to match the sorted manifest.

Inputs:
    - data/raw/metadata.json
    - data/processed/emotion_scores.csv
    - data/processed/temporal_features.csv
    - data/processed/place_ids.csv
    - data/processed/image_embeddings.npy + index
    - data/processed/text_embeddings.npy + index

Outputs:
    - data/processed/master_manifest.parquet
    - data/processed/final_image_vectors.npy
    - data/processed/final_text_vectors.npy
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import config from pipeline
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    EMOTION_SCORES_PATH,
    TEMPORAL_FEATURES_PATH,
    PLACE_IDS_PATH,
    IMAGE_EMBEDDINGS_PATH,
    IMAGE_EMBEDDING_INDEX_PATH,
    TEXT_EMBEDDINGS_PATH,
    CAPTION_EMBEDDING_INDEX_PATH,
    MASTER_MANIFEST_PATH,
    FINAL_IMAGE_VECTORS_PATH,
    FINAL_TEXT_VECTORS_PATH,
)


def load_data():
    """Load all input data files."""
    print("📂 Loading input files...")
    
    # 1. Metadata (Backbone)
    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)
    df_meta = pd.DataFrame(metadata["records"])
    print(f"   Metadata: {len(df_meta)} records")
    
    # 2. Features (CSVs)
    df_emotion = pd.read_csv(EMOTION_SCORES_PATH)
    df_temporal = pd.read_csv(TEMPORAL_FEATURES_PATH)
    df_place = pd.read_csv(PLACE_IDS_PATH)
    
    print(f"   Emotion: {len(df_emotion)} rows")
    print(f"   Temporal: {len(df_temporal)} rows")
    print(f"   Place: {len(df_place)} rows")
    
    # 3. Vectors (NPY + JSON)
    img_vecs = np.load(IMAGE_EMBEDDINGS_PATH)
    with open(IMAGE_EMBEDDING_INDEX_PATH) as f:
        img_idx = json.load(f)
        
    txt_vecs = np.load(TEXT_EMBEDDINGS_PATH)
    with open(CAPTION_EMBEDDING_INDEX_PATH) as f:
        txt_idx = json.load(f)
        
    print(f"   Image Vectors: {img_vecs.shape}")
    print(f"   Text Vectors: {txt_vecs.shape}")
    
    return df_meta, df_emotion, df_temporal, df_place, img_vecs, img_idx, txt_vecs, txt_idx


def merge_manifest(df_meta, df_emotion, df_temporal, df_place):
    """Merge all dataframes into a single master manifest."""
    print("\n🔗 Merging manifest...")
    
    # Ensure ID columns are consistent (int)
    df_meta["id"] = df_meta["id"].astype(int)
    df_emotion["id"] = df_emotion["id"].astype(int)
    df_temporal["id"] = df_temporal["id"].astype(int)
    df_place["id"] = df_place["id"].astype(int)
    
    # Merge
    # Start with metadata
    df_master = df_meta.copy()
    
    # Merge Emotion (drop user_id to avoid duplicates)
    df_master = df_master.merge(
        df_emotion.drop(columns=["user_id"], errors="ignore"),
        on="id",
        how="left"
    )
    
    # Merge Temporal
    df_master = df_master.merge(
        df_temporal.drop(columns=["user_id"], errors="ignore"),
        on="id",
        how="left"
    )
    
    # Merge Place
    df_master = df_master.merge(
        df_place.drop(columns=["user_id", "raw_lat", "raw_lon"], errors="ignore"),
        on="id",
        how="left"
    )
    
    # Sort by ID
    df_master = df_master.sort_values("id").reset_index(drop=True)
    
    print(f"   Master Manifest: {len(df_master)} rows")
    return df_master


def align_vectors(df_master, vectors, index_map, vector_name):
    """
    Align vectors to match the order of the master manifest.
    
    Args:
        df_master: Sorted master dataframe
        vectors: Raw numpy array of vectors
        index_map: Dict mapping record_id (str) -> {embedding_index: int}
        vector_name: Name for logging
        
    Returns:
        Aligned numpy array of vectors
    """
    print(f"\n📐 Aligning {vector_name}...")
    
    aligned_vectors = []
    dim = vectors.shape[1]
    missing_count = 0
    
    for _, row in df_master.iterrows():
        record_id = str(row["id"])
        
        if record_id in index_map:
            idx = index_map[record_id]["embedding_index"]
            vec = vectors[idx]
        else:
            # Handle missing vectors (e.g. text for redacted captions)
            # Use zero vector or NaN? Zero vector is safer for matrix ops, 
            # but we should track that it's missing.
            # For now, let's use Zero vector.
            vec = np.zeros(dim)
            missing_count += 1
            
        aligned_vectors.append(vec)
    
    aligned_vectors = np.stack(aligned_vectors)
    
    if missing_count > 0:
        print(f"   ⚠️  {missing_count} records missing {vector_name} (filled with zeros)")
    
    print(f"   Aligned Shape: {aligned_vectors.shape}")
    return aligned_vectors


def normalize_columns(df):
    """Normalize specific columns."""
    print("\n⚖️  Normalizing columns...")
    
    # Place ID to categorical
    if "place_id" in df.columns:
        df["place_id"] = df["place_id"].astype("category")
        print("   Converted place_id to categorical")
        
    # Valence/Arousal are already 0-1 from the extraction script?
    # Let's check ranges just in case, but usually they are.
    # If they are not, we should scale them. 
    # Based on plan.md, they are probabilities or 0-1 scores.
    
    return df


def main():
    # 1. Load Data
    (df_meta, df_emotion, df_temporal, df_place, 
     img_vecs, img_idx, txt_vecs, txt_idx) = load_data()
    
    # 2. Merge Manifest
    df_master = merge_manifest(df_meta, df_emotion, df_temporal, df_place)
    
    # 3. Normalize
    df_master = normalize_columns(df_master)
    
    # 4. Align Vectors
    final_img_vecs = align_vectors(df_master, img_vecs, img_idx, "Image Vectors")
    final_txt_vecs = align_vectors(df_master, txt_vecs, txt_idx, "Text Vectors")
    
    # 5. Save Outputs
    print("\n💾 Saving outputs...")
    
    # Save Parquet
    df_master.to_parquet(MASTER_MANIFEST_PATH, index=False)
    print(f"   Manifest: {MASTER_MANIFEST_PATH}")
    
    # Save Vectors
    np.save(FINAL_IMAGE_VECTORS_PATH, final_img_vecs)
    print(f"   Image Vectors: {FINAL_IMAGE_VECTORS_PATH}")
    
    np.save(FINAL_TEXT_VECTORS_PATH, final_txt_vecs)
    print(f"   Text Vectors: {FINAL_TEXT_VECTORS_PATH}")
    
    print("\n✅ Grand Fusion Complete!")


if __name__ == "__main__":
    main()
