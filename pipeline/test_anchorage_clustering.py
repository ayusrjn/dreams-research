import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import SENTENCE_BERT_MODEL, PROCESSED_DIR, LOCATION_COLLECTION_NAME
from db import get_collection
from location_semantic import reverse_geocode, generate_description, get_gemini_api_key, get_nominatim_user_agent
from location_embeddings import get_caption, store_in_chromadb, cluster_embeddings

TEST_DIR = Path(__file__).parent.parent / "data" / "raw" / "test_anchorage"
CACHE_DIR = PROCESSED_DIR / "anchorage_cache"

TEST_RECORDS = [
    {"id": "park_earthquake", "category": "park", "lat": 61.1958, "lon": -149.9642, "image": "park_earthquake.png"},
    {"id": "park_kincaid", "category": "park", "lat": 61.1537, "lon": -150.0564, "image": "park_kincaid.png"},
    {"id": "park_westchester", "category": "park", "lat": 61.2058, "lon": -149.9141, "image": "park_westchester.png"},
    {"id": "hosp_regional", "category": "hospital", "lat": 61.1862, "lon": -149.8783, "image": "hospital_regional.png"},
    {"id": "hosp_providence", "category": "hospital", "lat": 61.1880, "lon": -149.8203, "image": "hospital_providence.png"},
    {"id": "hosp_native", "category": "hospital", "lat": 61.2116, "lon": -149.8259, "image": "hospital_native.png"},
    {"id": "rest_moose", "category": "restaurant", "lat": 61.1903, "lon": -149.8684, "image": "restaurant_moose.png"},
    {"id": "rest_snowcity", "category": "restaurant", "lat": 61.2173, "lon": -149.8884, "image": "restaurant_snowcity.png"},
    {"id": "rest_clubparis", "category": "restaurant", "lat": 61.2177, "lon": -149.8867, "image": "restaurant_clubparis.png"},
    {"id": "res_south", "category": "residential", "lat": 61.2100, "lon": -149.9000, "image": "residential_south.png"},
    {"id": "res_turnagain", "category": "residential", "lat": 61.1949, "lon": -149.9380, "image": "residential_turnagain.png"},
    {"id": "res_rogers", "category": "residential", "lat": 61.2000, "lon": -149.8500, "image": "residential_rogers.png"}
]

def load_cached(rid):
    p = CACHE_DIR / f"{rid}.json"
    return json.load(open(p)) if p.exists() else None

def retry(func, *args, **kwargs):
    for attempt in range(4):
        try: return func(*args, **kwargs)
        except Exception as e:
            if "quota" in str(e).lower() and "exceeded" in str(e).lower(): raise e
            if attempt < 3:
                time.sleep(25 * (2 ** attempt))
                continue
            raise

async def process(rec, key, ua):
    if cached := load_cached(rec["id"]): return cached
    
    img_path = TEST_DIR / rec["image"]
    if not img_path.exists(): return None

    try: caption = retry(get_caption, str(img_path), key)
    except Exception: caption = None

    time.sleep(2)
    try: geocode = await reverse_geocode(rec["lat"], rec["lon"], user_agent=ua)
    except Exception: geocode = {"display_name": None, "address": None, "raw": None}

    time.sleep(5)
    try: desc = retry(generate_description, rec["lat"], rec["lon"], geocode, api_key=key, caption=caption)
    except Exception: desc = None

    if not desc: return None

    res = {"id": rec["id"], "category": rec["category"], "lat": rec["lat"], "lon": rec["lon"], "caption": caption or "", "geocode_display_name": geocode.get("display_name") or "(unknown)", "description": desc}
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{rec['id']}.json", "w") as f:
        json.dump(res, f, indent=2)

    time.sleep(5)
    return res

def eval_clusters(res, lbls):
    cats, pure_c = {}, {}
    for r, l in zip(res, lbls):
        cats.setdefault(r["category"], {"lbls": set(), "recs": []})["lbls"].add(int(l))
        cats[r["category"]]["recs"].append(f"{r['id']}(c{l})")
        pure_c.setdefault(int(l), set()).add(r["category"])
        
    purity = sum(1 for c in pure_c.values() if len(c) == 1) / len(pure_c) if pure_c else 0.0
    return {"total_records": len(res), "total_clusters": len(pure_c), "cluster_purity": round(purity, 2), "categories": {c: {"records": v["recs"], "unique_cluster_labels": sorted(v["lbls"]), "homogeneous": len(v["lbls"]) == 1 and -1 not in v["lbls"]} for c, v in cats.items()}}

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--cached", action="store_true")
    args = args.parse_args()

    key, ua = get_gemini_api_key(), get_nominatim_user_agent()

    results = []
    for rec in TEST_RECORDS:
        if args.cached:
            if c := load_cached(rec["id"]): results.append(c)
        elif res := asyncio.run(process(rec, key, ua)):
            results.append(res)

    if not results: sys.exit(1)

    sbert = SentenceTransformer(SENTENCE_BERT_MODEL)
    emb = sbert.encode([r["description"] for r in results], convert_to_numpy=True, normalize_embeddings=True)

    store_in_chromadb(get_collection(LOCATION_COLLECTION_NAME), results, emb)
    labels = cluster_embeddings(emb, results, eps=0.35, min_samples=2)

    report = eval_clusters(results, labels)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "anchorage_test_report.json", "w") as f:
        json.dump({"report": report, "results": results, "labels": labels.tolist()}, f, indent=2)

if __name__ == "__main__":
    main()
