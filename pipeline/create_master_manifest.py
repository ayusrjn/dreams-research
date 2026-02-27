import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME
from db import init_db, get_collection

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    conn = init_db()
    
    for table in ["memories", "emotion_scores", "temporal_features", "location_descriptions", "master_manifest"]:
        print(f"{table}: {conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0]}")

    for col in ["valence", "arousal", "sin_hour"]:
        print(f"{col} NULLs: {conn.execute(f'SELECT count(*) FROM master_manifest WHERE {col} IS NULL').fetchone()[0]}")

    for coll in [IMAGE_COLLECTION_NAME, CAPTION_COLLECTION_NAME, LOCATION_COLLECTION_NAME]:
        print(f"{coll}: {get_collection(coll).count()} vectors")

    row = conn.execute("SELECT * FROM master_manifest LIMIT 1").fetchone()
    if row:
        for c, v in zip([d[0] for d in conn.execute("SELECT * FROM master_manifest LIMIT 0").description], row):
            if v is not None:
                print(f"{c}: {v}")

    if args.export:
        import pandas as pd
        pd.read_sql_query("SELECT * FROM master_manifest", conn).to_parquet(PROCESSED_DIR / "master_manifest.parquet", index=False)

    conn.close()

if __name__ == "__main__":
    main()
