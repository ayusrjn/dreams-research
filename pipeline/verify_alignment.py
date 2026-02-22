#!/usr/bin/env python3
"""
Verification Script for Grand Fusion

Checks consistency between SQLite database and ChromaDB vector collections.
"""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    SENTENCE_BERT_MODEL,
    IMAGE_COLLECTION_NAME,
    CAPTION_COLLECTION_NAME,
    LOCATION_COLLECTION_NAME,
)
from db import init_db, get_collection


def main():
    print("=" * 60)
    print("DREAMS Research - Grand Fusion Verification")
    print("=" * 60)
    
    # 1. Load SQLite
    print("\n[INFO] Loading database...")
    conn = init_db()
    
    memory_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    manifest_count = conn.execute("SELECT count(*) FROM master_manifest").fetchone()[0]
    
    print(f"   Memories: {memory_count} rows")
    print(f"   Master Manifest (VIEW): {manifest_count} rows")
    
    if memory_count == 0:
        print("\n[WARN] No records in database. Run the pipeline first.")
        conn.close()
        return
    
    # 2. Length Checks
    print("\n[INFO] Checking table counts vs memories...")
    tables = {
        "emotion_scores": conn.execute("SELECT count(*) FROM emotion_scores").fetchone()[0],
        "temporal_features": conn.execute("SELECT count(*) FROM temporal_features").fetchone()[0],
    }
    
    for table, count in tables.items():
        status = "[OK]" if count > 0 else "[WARN]"
        print(f"   {status} {table}: {count}")
    
    # 3. ChromaDB consistency
    print("\n[INFO] Checking ChromaDB collections...")
    for coll_name in [IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME]:
        collection = get_collection(coll_name)
        coll_count = collection.count()
        status = "[OK]" if coll_count > 0 else "[WARN]"
        print(f"   {status} {coll_name}: {coll_count} vectors")
    
    # 4. Semantic Sanity Check
    print("\n[INFO] Semantic Sanity Check (first caption)...")
    caption_collection = get_collection(CAPTION_COLLECTION_NAME)
    
    if caption_collection.count() > 0:
        # Get first record with a caption from SQLite
        row = conn.execute(
            "SELECT id, caption FROM memories WHERE caption IS NOT NULL AND caption != '' LIMIT 1"
        ).fetchone()
        
        if row:
            record_id = str(row[0])
            caption = row[1]
            
            print(f"   Record ID: {record_id}")
            print(f"   Caption: '{caption[:80]}'")
            
            # Get stored vector from ChromaDB
            stored = caption_collection.get(
                ids=[record_id],
                include=["embeddings"]
            )
            
            if stored["ids"]:
                stored_vec = stored["embeddings"][0]
                
                # Re-encode caption
                print("   Loading model for similarity check...")
                model = SentenceTransformer(SENTENCE_BERT_MODEL)
                ref_vec = model.encode([caption], normalize_embeddings=True)[0]
                
                # Compare
                sim = cosine_similarity([stored_vec], [ref_vec])[0][0]
                print(f"   Cosine Similarity (Stored vs Re-computed): {sim:.4f}")
                
                if sim > 0.99:
                    print("   [OK] Vector matches caption content!")
                else:
                    print("   [ERROR] Vector mismatch!")
            else:
                print(f"   [WARN] Record {record_id} not found in ChromaDB")
        else:
            print("   [WARN] No captions found in database")
    else:
        print("   [WARN] Caption collection is empty")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("[OK] Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
