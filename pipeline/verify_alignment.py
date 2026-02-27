import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from config import SENTENCE_BERT_MODEL, IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME
from db import init_db, get_collection

def main():
    conn = init_db()
    
    mem_count = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
    man_count = conn.execute("SELECT count(*) FROM master_manifest").fetchone()[0]
    print(f"Memories: {mem_count} rows\nMaster Manifest: {man_count} rows")
    
    if mem_count == 0:
        conn.close()
        return

    for t in ("emotion_scores", "temporal_features"):
        print(f"{t}: {conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0]}")

    for c in (IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME):
        print(f"{c}: {get_collection(c).count()} vectors")

    cap_coll = get_collection(CAPTION_COLLECTION_NAME)
    if cap_coll.count() > 0:
        if row := conn.execute("SELECT id, caption FROM memories WHERE caption IS NOT NULL AND caption != '' LIMIT 1").fetchone():
            rid, cap = str(row[0]), row[1]
            if stored := cap_coll.get(ids=[rid], include=["embeddings"])["embeddings"]:
                sim = cosine_similarity([stored[0]], [SentenceTransformer(SENTENCE_BERT_MODEL).encode([cap], normalize_embeddings=True)[0]])[0][0]
                print(f"Cosine Similarity (Stored vs Re-computed): {sim:.4f}")

    conn.close()

if __name__ == "__main__":
    main()
