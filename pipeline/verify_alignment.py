#!/usr/bin/env python3
"""
Verification Script for Grand Fusion

Checks alignment between Master Manifest and Vector Files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MASTER_MANIFEST_PATH,
    FINAL_IMAGE_VECTORS_PATH,
    FINAL_TEXT_VECTORS_PATH,
    SENTENCE_BERT_MODEL,
)


def main():
    print("=" * 60)
    print("DREAMS Research - Grand Fusion Verification")
    print("=" * 60)
    
    # 1. Load Files
    print("\n📂 Loading files...")
    if not MASTER_MANIFEST_PATH.exists():
        print(f"❌ Manifest not found: {MASTER_MANIFEST_PATH}")
        return
        
    df = pd.read_parquet(MASTER_MANIFEST_PATH)
    
    # Explicitly check for vector files
    if not FINAL_IMAGE_VECTORS_PATH.exists():
        print(f"❌ Missing vector file FINAL_IMAGE_VECTORS_PATH: {FINAL_IMAGE_VECTORS_PATH}")
        sys.exit(1)
        
    if not FINAL_TEXT_VECTORS_PATH.exists():
        print(f"❌ Missing vector file FINAL_TEXT_VECTORS_PATH: {FINAL_TEXT_VECTORS_PATH}")
        sys.exit(1)
        
    img_vecs = np.load(FINAL_IMAGE_VECTORS_PATH)
    txt_vecs = np.load(FINAL_TEXT_VECTORS_PATH)
    
    print(f"   Manifest: {len(df)} rows")
    print(f"   Image Vecs: {img_vecs.shape}")
    print(f"   Text Vecs: {txt_vecs.shape}")
    
    # 2. Length Check
    print("\n📏 Checking lengths...")
    if len(df) == len(img_vecs) == len(txt_vecs):
        print("   ✅ Lengths match!")
    else:
        print("   ❌ Length mismatch!")
        print(f"      DF: {len(df)}")
        print(f"      Img: {len(img_vecs)}")
        print(f"      Txt: {len(txt_vecs)}")
        sys.exit(1)
        
    # 3. Semantic Sanity Check
    print("\n🧠 Semantic Sanity Check (ID=0 / First Row)...")
    
    # Get first row
    row = df.iloc[0]
    record_id = row["id"]
    caption = row["caption"]
    
    print(f"   Record ID: {record_id}")
    print(f"   Caption: '{caption}'")
    
    # Get vector
    vec = txt_vecs[0]
    
    # Check if vector is zero (missing)
    if np.all(vec == 0):
        print("   ⚠️  Vector is all zeros (likely missing/redacted)")
    else:
        # Load model to verify
        print("   Loading model for similarity check...")
        model = SentenceTransformer(SENTENCE_BERT_MODEL)
        
        # Encode caption again
        ref_vec = model.encode([caption])[0]
        
        # Compute similarity
        sim = cosine_similarity([vec], [ref_vec])[0][0]
        print(f"   Cosine Similarity (Stored vs Re-computed): {sim:.4f}")
        
        if sim > 0.99:
            print("   ✅ Vector matches caption content!")
        else:
            print("   ❌ Vector mismatch!")
            
    print("\n" + "=" * 60)
    print("✅ Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
