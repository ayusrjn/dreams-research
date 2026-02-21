#!/usr/bin/env python3
"""
Anchorage Alaska – Location Embeddings Clustering Test

Processes 12 generated images of Anchorage locations (4 categories × 3 each)
through the full location-embeddings pipeline and checks whether semantically
similar places cluster together.

Features:
    - Per-record caching: successful results are saved to disk so re-runs
      only call the API for records not yet processed. This is critical for
      free-tier Gemini quotas (20 requests/day).
    - Exponential backoff on rate-limit errors (429/503).
    - Inter-request delays to respect rate limits.

Usage:
    python pipeline/test_anchorage_clustering.py           # process & cluster
    python pipeline/test_anchorage_clustering.py --cached   # skip API, cluster cached results only
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SENTENCE_BERT_MODEL,
    CHROMA_DB_DIR,
    PROCESSED_DIR,
    LOCATION_COLLECTION_NAME,
)
from location_semantic import (
    reverse_geocode,
    generate_description,
    get_gemini_api_key,
    get_nominatim_user_agent,
)
from location_embeddings import (
    get_image_caption,
    get_chroma_collection,
    store_in_chromadb,
    cluster_embeddings,
    embed_descriptions,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DIR = Path(__file__).parent.parent / "data" / "raw" / "test_anchorage"
CACHE_DIR = PROCESSED_DIR / "anchorage_cache"


# ---------------------------------------------------------------------------
# Test data: 12 Anchorage locations with real GPS coordinates
# ---------------------------------------------------------------------------

TEST_RECORDS = [
    # Parks / Nature
    {"id": "park_earthquake",   "category": "park",       "lat": 61.1958, "lon": -149.9642, "image": "park_earthquake.png"},
    {"id": "park_kincaid",      "category": "park",       "lat": 61.1537, "lon": -150.0564, "image": "park_kincaid.png"},
    {"id": "park_westchester",  "category": "park",       "lat": 61.2058, "lon": -149.9141, "image": "park_westchester.png"},
    # Hospitals / Medical
    {"id": "hosp_regional",     "category": "hospital",   "lat": 61.1862, "lon": -149.8783, "image": "hospital_regional.png"},
    {"id": "hosp_providence",   "category": "hospital",   "lat": 61.1880, "lon": -149.8203, "image": "hospital_providence.png"},
    {"id": "hosp_native",       "category": "hospital",   "lat": 61.2116, "lon": -149.8259, "image": "hospital_native.png"},
    # Restaurants / Cafés
    {"id": "rest_moose",        "category": "restaurant", "lat": 61.1903, "lon": -149.8684, "image": "restaurant_moose.png"},
    {"id": "rest_snowcity",     "category": "restaurant", "lat": 61.2173, "lon": -149.8884, "image": "restaurant_snowcity.png"},
    {"id": "rest_clubparis",    "category": "restaurant", "lat": 61.2177, "lon": -149.8867, "image": "restaurant_clubparis.png"},
    # Residential
    {"id": "res_south",         "category": "residential","lat": 61.2100, "lon": -149.9000, "image": "residential_south.png"},
    {"id": "res_turnagain",     "category": "residential","lat": 61.1949, "lon": -149.9380, "image": "residential_turnagain.png"},
    {"id": "res_rogers",        "category": "residential","lat": 61.2000, "lon": -149.8500, "image": "residential_rogers.png"},
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(record_id: str) -> Path:
    return CACHE_DIR / f"{record_id}.json"


def load_cached(record_id: str) -> dict | None:
    p = _cache_path(record_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_to_cache(result: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(result["id"]), "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Gemini retry helper
# ---------------------------------------------------------------------------

def _retry_gemini(func, *args, max_retries=3, base_wait=25, **kwargs):
    """Call a Gemini function with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or
                    "503" in err_str or "UNAVAILABLE" in err_str):
                if attempt < max_retries:
                    wait = base_wait * (2 ** attempt)
                    print(f"      ⏳ Rate limited, waiting {wait}s (retry {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
            raise


# ---------------------------------------------------------------------------
# Single-record pipeline
# ---------------------------------------------------------------------------

async def process_record(rec: dict, api_key: str, user_agent: str) -> dict | None:
    """Run caption → geocode → describe for one record. Caches result on success."""

    # Check cache first
    cached = load_cached(rec["id"])
    if cached:
        print(f"   📦 {rec['id']} [{rec['category']}] (cached)")
        print(f"      Description: {cached['description']}")
        return cached

    image_path = TEST_DIR / rec["image"]
    if not image_path.exists():
        print(f"   ⚠️  {rec['id']}: image {image_path} not found (skipped)")
        return None

    lat, lon = rec["lat"], rec["lon"]

    # 1) Caption (with retry)
    try:
        caption = _retry_gemini(get_image_caption, str(image_path), api_key)
    except Exception as e:
        print(f"   ⚠️  {rec['id']}: captioning failed ({e})")
        caption = None

    # Delay: respect Nominatim 1 req/sec + spread Gemini calls
    time.sleep(2)

    # 2) Geocode
    try:
        geocode_data = await reverse_geocode(lat, lon, user_agent=user_agent)
    except Exception as e:
        print(f"   ⚠️  {rec['id']}: geocoding failed ({e})")
        geocode_data = {"display_name": None, "address": None, "raw": None}

    # Delay before next Gemini call
    time.sleep(5)

    # 3) Description (with retry)
    try:
        description = _retry_gemini(
            generate_description,
            lat, lon, geocode_data, api_key=api_key, caption=caption
        )
    except Exception as e:
        print(f"   ⚠️  {rec['id']}: description failed ({e})")
        description = None

    if not description:
        print(f"   ⚠️  {rec['id']}: no description produced (skipped)")
        return None

    display_name = geocode_data.get("display_name") or "(unknown)"
    print(f"   ✅ {rec['id']} [{rec['category']}]")
    print(f"      Location:    {display_name}")
    print(f"      Caption:     {caption}")
    print(f"      Description: {description}")

    result = {
        "id": rec["id"],
        "category": rec["category"],
        "lat": lat,
        "lon": lon,
        "caption": caption or "",
        "geocode_display_name": display_name,
        "description": description,
    }

    # Save to cache immediately so we don't lose progress
    save_to_cache(result)

    # Delay before processing next record (rate limit breathing room)
    time.sleep(5)

    return result


# ---------------------------------------------------------------------------
# Cluster evaluation
# ---------------------------------------------------------------------------

def evaluate_clusters(results: list[dict], labels: np.ndarray) -> dict:
    """Check if same-category records land in the same cluster."""

    cat_to_labels: dict[str, set[int]] = {}
    cat_to_records: dict[str, list[str]] = {}
    for r, lbl in zip(results, labels):
        cat = r["category"]
        cat_to_labels.setdefault(cat, set()).add(int(lbl))
        cat_to_records.setdefault(cat, []).append(f"{r['id']}(c{lbl})")

    cluster_to_cats: dict[int, set[str]] = {}
    for r, lbl in zip(results, labels):
        lbl_int = int(lbl)
        cluster_to_cats.setdefault(lbl_int, set()).add(r["category"])

    pure = sum(1 for cats in cluster_to_cats.values() if len(cats) == 1)
    total = len(cluster_to_cats)
    purity = pure / total if total > 0 else 0.0

    report = {
        "total_records": len(results),
        "total_clusters": total,
        "cluster_purity": round(purity, 2),
        "categories": {},
    }

    for cat in sorted(cat_to_labels):
        lbls = cat_to_labels[cat]
        records = cat_to_records[cat]
        homogeneous = len(lbls) == 1 and -1 not in lbls
        report["categories"][cat] = {
            "records": records,
            "unique_cluster_labels": sorted(lbls),
            "homogeneous": homogeneous,
        }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true",
                        help="Skip API calls; only use cached results.")
    args = parser.parse_args()

    print("=" * 65)
    print(" ANCHORAGE ALASKA – LOCATION CLUSTERING TEST")
    print("=" * 65)
    print()

    api_key = get_gemini_api_key()
    user_agent = get_nominatim_user_agent()

    # --- Check cache status ---
    cached_ids = set()
    for rec in TEST_RECORDS:
        if load_cached(rec["id"]):
            cached_ids.add(rec["id"])
    remaining = len(TEST_RECORDS) - len(cached_ids)
    print(f"   Cache: {len(cached_ids)}/{len(TEST_RECORDS)} records cached")
    print(f"   Remaining API calls needed: ~{remaining * 2} (caption + description each)\n")

    # --- Step 1: Process records ---
    print("📷  Step 1/5: Processing records (caption → geocode → describe)\n")
    results = []
    for i, rec in enumerate(TEST_RECORDS):
        print(f"\n   --- Record {i+1}/{len(TEST_RECORDS)} ---")
        if args.cached:
            cached = load_cached(rec["id"])
            if cached:
                print(f"   📦 {rec['id']} [{rec['category']}] (cached)")
                print(f"      Description: {cached['description']}")
                results.append(cached)
            else:
                print(f"   ⏭️  {rec['id']}: not cached (skipping in --cached mode)")
        else:
            result = asyncio.run(process_record(rec, api_key, user_agent))
            if result:
                results.append(result)

    if not results:
        print("\n⚠️  No records produced descriptions. Aborting.")
        sys.exit(1)

    print(f"\n   Processed: {len(results)}/{len(TEST_RECORDS)}")

    # --- Step 2: Embed ---
    print(f"\n🧠  Step 2/5: Embedding ({SENTENCE_BERT_MODEL})...")
    sbert = SentenceTransformer(SENTENCE_BERT_MODEL)
    descriptions = [r["description"] for r in results]
    embeddings = embed_descriptions(descriptions, sbert)
    print(f"   Shape: {embeddings.shape}")

    # --- Step 3: ChromaDB ---
    print(f"\n💾  Step 3/5: Storing in ChromaDB ({LOCATION_COLLECTION_NAME})...")
    collection = get_chroma_collection(LOCATION_COLLECTION_NAME)
    store_in_chromadb(collection, results, embeddings)
    print(f"   Collection size: {collection.count()}")

    # --- Step 4: Cluster ---
    print(f"\n🔗  Step 4/5: DBSCAN clustering (eps=0.35, min_samples=2)...")
    labels = cluster_embeddings(embeddings, results, eps=0.35, min_samples=2)

    # --- Step 5: Evaluate ---
    print(f"\n📊  Step 5/5: Evaluating cluster quality...\n")
    report = evaluate_clusters(results, labels)

    print("   " + "-" * 55)
    print(f"   Cluster purity: {report['cluster_purity']:.0%}")
    print(f"   Total clusters: {report['total_clusters']}")
    print("   " + "-" * 55)

    for cat, info in report["categories"].items():
        tag = "✅" if info["homogeneous"] else "⚠️ "
        print(f"   {tag} {cat:12s} → clusters {info['unique_cluster_labels']}  records: {info['records']}")

    # --- Save report ---
    output_path = PROCESSED_DIR / "anchorage_test_report.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    save_data = {
        "report": report,
        "results": results,
        "labels": labels.tolist(),
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n   Full report saved: {output_path}")

    # --- Similarity matrix ---
    print("\n📐  Cosine similarity matrix:\n")
    sim = np.dot(embeddings, embeddings.T)
    ids = [r["id"] for r in results]

    # Header
    hdr = "              " + " ".join(f"{i:>13s}" for i in ids)
    print(hdr)
    for i, row in enumerate(sim):
        vals = " ".join(f"{v:13.2f}" for v in row)
        print(f"   {ids[i]:>13s} {vals}")

    print("\n" + "=" * 65)
    print(" TEST COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
