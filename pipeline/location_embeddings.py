#!/usr/bin/env python3
"""
Location Embeddings Pipeline

Processes memory records through:
1. Image captioning (Gemini vision)
2. Reverse geocoding (Nominatim)
3. Semantic description generation (Gemini text)
4. Description embedding (Sentence-BERT all-MiniLM-L6-v2)
5. Persistent storage (ChromaDB)
6. Semantic clustering (DBSCAN on cosine distance)

Usage:
    python pipeline/location_embeddings.py --batch                # process all records
    python pipeline/location_embeddings.py --image <path> --lat <float> --lon <float>
    python pipeline/location_embeddings.py --query "residential area near hospital"
"""

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent))

from google import genai
from config import (
    RAW_IMAGES_DIR,
    RAW_METADATA_PATH,
    SENTENCE_BERT_MODEL,
    LOCATION_COLLECTION_NAME,
)
from db import get_collection
from location_semantic import (
    reverse_geocode,
    generate_description,
    get_gemini_api_key,
    get_nominatim_user_agent,
)


# ---------------------------------------------------------------------------
# 1. Image captioning (Gemini Vision)
# ---------------------------------------------------------------------------

def get_image_caption(image_path: str, api_key: str) -> str:
    """Generate a short caption for an image using Gemini vision model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    image_bytes = image_path.read_bytes()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                    {
                        "text": (
                            "You are a visual scene classification assistant.\n\n"
                            "Describe the environment shown in this image in ONE concise sentence (max 25 words).\n\n"
                            "Focus only on:\n"
                            "- Type of place (e.g., hospital, church, restaurant, park, residential building)\n"
                            "- Setting (urban, rural, indoor, outdoor)\n"
                            "- Functional purpose if identifiable\n\n"
                            "Do NOT describe emotions, artistic style, or speculation.\n"
                            "Be factual and precise.\n"
                            "Return only the sentence."
                        ),
                    },
                ],
            }
        ],
    )

    return response.text.strip()


# ---------------------------------------------------------------------------
# 2. Single-record pipeline (caption → geocode → describe)
# ---------------------------------------------------------------------------

async def process_single_record(
    record: dict,
    api_key: str,
    user_agent: str,
) -> dict | None:
    """Process one metadata record through the full pipeline.

    Returns a result dict or None if the record cannot be processed.
    """
    record_id = record.get("id")
    lat = record.get("lat")
    lon = record.get("lon")

    if lat is None or lon is None:
        print(f"   ⚠️  Record {record_id}: No coordinates (skipped)")
        return None

    lat, lon = float(lat), float(lon)

    # --- Find image on disk ---
    local_rel = record.get("local_image")
    image_path = None
    if local_rel:
        candidate = RAW_IMAGES_DIR / local_rel
        if candidate.exists():
            image_path = str(candidate)

    if image_path is None:
        for ext in ("jpg", "jpeg", "png", "webp"):
            candidate = RAW_IMAGES_DIR / f"{record_id}.{ext}"
            if candidate.exists():
                image_path = str(candidate)
                break

    if image_path is None:
        print(f"   ⚠️  Record {record_id}: Image not found (skipped)")
        return None

    # --- Caption ---
    try:
        caption = get_image_caption(image_path, api_key)
    except Exception as e:
        print(f"   ⚠️  Record {record_id}: Captioning failed ({e})")
        caption = None

    # --- Geocode ---
    try:
        geocode_data = await reverse_geocode(lat, lon, user_agent=user_agent)
    except Exception as e:
        print(f"   ⚠️  Record {record_id}: Geocoding failed ({e})")
        geocode_data = {"display_name": None, "address": None, "raw": None}

    # --- Description ---
    try:
        description = generate_description(
            lat, lon, geocode_data, api_key=api_key, caption=caption
        )
    except Exception as e:
        print(f"   ⚠️  Record {record_id}: Description generation failed ({e})")
        description = None

    if not description:
        print(f"   ⚠️  Record {record_id}: No description produced (skipped)")
        return None

    display_name = geocode_data.get("display_name") or "(unknown)"
    print(f"   ✅ Record {record_id}: {display_name}")
    print(f"      Caption:     {caption}")
    print(f"      Description: {description}")

    return {
        "id": str(record_id),
        "lat": lat,
        "lon": lon,
        "caption": caption or "",
        "geocode_display_name": display_name,
        "description": description,
    }


# ---------------------------------------------------------------------------
# 3. Embedding
# ---------------------------------------------------------------------------

def embed_descriptions(
    descriptions: list[str], model: SentenceTransformer
) -> np.ndarray:
    """Batch-encode descriptions into normalized embeddings."""
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# 4. ChromaDB storage
# ---------------------------------------------------------------------------

def store_in_chromadb(
    collection,
    results: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Upsert records into ChromaDB with pre-computed embeddings."""
    ids = [r["id"] for r in results]
    documents = [r["description"] for r in results]
    metadatas = [
        {
            "lat": r["lat"],
            "lon": r["lon"],
            "caption": r["caption"],
            "geocode_display_name": r["geocode_display_name"],
        }
        for r in results
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )


# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    results: list[dict],
    eps: float = 0.3,
    min_samples: int = 2,
) -> np.ndarray:
    """Run DBSCAN clustering on cosine distance.

    Args:
        embeddings: Normalized embedding matrix (N, D).
        results: Corresponding result dicts (for printing).
        eps: Maximum cosine distance between two samples in the same cluster.
        min_samples: Minimum cluster size.

    Returns:
        Array of cluster labels (-1 = noise / unclustered).
    """
    distance_matrix = 1.0 - np.dot(embeddings, embeddings.T)
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    )
    labels = clustering.fit_predict(distance_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    print(f"   Clusters found: {n_clusters}")
    print(f"   Unclustered (noise): {n_noise}")

    for cluster_id in sorted(set(labels)):
        members = [
            results[i] for i, l in enumerate(labels) if l == cluster_id
        ]
        tag = f"Cluster {cluster_id}" if cluster_id >= 0 else "Unclustered"
        print(f"\n   [{tag}]")
        for m in members:
            print(f"     - Record {m['id']}: {m['description'][:80]}")

    return labels


# ---------------------------------------------------------------------------
# 6. Query demo
# ---------------------------------------------------------------------------

def query_similar(
    collection,
    query_text: str,
    model: SentenceTransformer,
    n_results: int = 5,
) -> None:
    """Find and display the most semantically similar stored descriptions."""
    query_embedding = model.encode(
        [query_text], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    print(f"\n   Query: \"{query_text}\"")
    print(f"   Top {len(results['ids'][0])} results:\n")

    for i, (doc_id, doc, meta, dist) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        similarity = 1.0 - dist
        print(f"   {i+1}. [Record {doc_id}] (similarity: {similarity:.4f})")
        print(f"      Location:    {meta.get('geocode_display_name', 'N/A')}")
        print(f"      Description: {doc}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Location Embeddings Pipeline (test script)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--batch",
        action="store_true",
        help="Process all records from metadata.json.",
    )
    mode.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query ChromaDB for similar location descriptions.",
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument(
        "--eps",
        type=float,
        default=0.3,
        help="DBSCAN cosine-distance epsilon (default: 0.3).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples (default: 2).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("DREAMS Research - Location Embeddings Pipeline")
    print("=" * 60)
    print()

    args = parse_args()

    # --- Query mode -------------------------------------------------------
    if args.query:
        print("🔍 Query Mode")
        print(f"   Loading Sentence-BERT model: {SENTENCE_BERT_MODEL}")
        sbert = SentenceTransformer(SENTENCE_BERT_MODEL)

        collection = get_collection(LOCATION_COLLECTION_NAME)
        if collection.count() == 0:
            print("   ChromaDB collection is empty. Run --batch first.")
            sys.exit(1)

        query_similar(collection, args.query, sbert)
        return

    # --- Single-image mode ------------------------------------------------
    if args.image and args.lat is not None and args.lon is not None:
        api_key = get_gemini_api_key()
        user_agent = get_nominatim_user_agent()

        unique_seed = f"{args.lat}:{args.lon}:{time.time()}"
        entry_id = hashlib.md5(unique_seed.encode()).hexdigest()[:12]

        record = {
            "id": entry_id,
            "lat": args.lat,
            "lon": args.lon,
            "local_image": None,
        }

        print("📷 Single-Image Mode\n")
        print(f"  [1/4] Captioning image: {args.image}")
        caption = get_image_caption(args.image, api_key)
        print(f"        Caption: {caption}")

        print(f"  [2/4] Reverse-geocoding ({args.lat}, {args.lon})...")
        geocode_data = asyncio.run(
            reverse_geocode(args.lat, args.lon, user_agent=user_agent)
        )
        display_name = geocode_data.get("display_name") or "(unknown)"
        print(f"        Location: {display_name}")

        print(f"  [3/4] Generating semantic description...")
        description = generate_description(
            args.lat, args.lon, geocode_data, api_key=api_key, caption=caption
        )
        print(f"        Description: {description}")

        print(f"  [4/4] Embedding + storing...")
        sbert = SentenceTransformer(SENTENCE_BERT_MODEL)
        embedding = sbert.encode(
            [description], normalize_embeddings=True
        )
        collection = get_collection(LOCATION_COLLECTION_NAME)
        store_in_chromadb(
            collection,
            [
                {
                    "id": entry_id,
                    "lat": args.lat,
                    "lon": args.lon,
                    "caption": caption,
                    "geocode_display_name": display_name,
                    "description": description,
                }
            ],
            embedding,
        )
        print("        Stored in ChromaDB.\n")

        print("=" * 60)
        print("Done!")
        print("=" * 60)
        return

    # --- Batch mode -------------------------------------------------------
    if not args.batch:
        print("No mode specified, defaulting to --batch.\n")

    api_key = get_gemini_api_key()
    user_agent = get_nominatim_user_agent()

    # Load metadata
    print("📂 Step 1: Loading metadata...")
    if not RAW_METADATA_PATH.exists():
        print(f"   Metadata not found: {RAW_METADATA_PATH}")
        print("   Run pull_data.py first.")
        sys.exit(1)

    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)

    records = metadata.get("records", [])
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records:  {len(records)}")

    # Process each record
    print(f"\n📷 Step 2: Processing records (caption → geocode → describe)...")
    results = []
    for i, rec in enumerate(records):
        print(f"\n   --- Record {i+1}/{len(records)} ---")
        result = asyncio.run(
            process_single_record(rec, api_key, user_agent)
        )
        if result:
            results.append(result)

    if not results:
        print("\n⚠️  No records produced descriptions. Check data & API key.")
        sys.exit(1)

    print(f"\n   Processed: {len(results)}/{len(records)} records")

    # Embed
    print(f"\n🧠 Step 3: Embedding descriptions ({SENTENCE_BERT_MODEL})...")
    sbert = SentenceTransformer(SENTENCE_BERT_MODEL)
    descriptions = [r["description"] for r in results]
    embeddings = embed_descriptions(descriptions, sbert)
    print(f"   Shape: {embeddings.shape}")

    # Store in ChromaDB
    print(f"\n💾 Step 4: Storing in ChromaDB ({LOCATION_COLLECTION_NAME})...")
    collection = get_collection(LOCATION_COLLECTION_NAME)
    store_in_chromadb(collection, results, embeddings)
    print(f"   Collection size: {collection.count()}")

    # Cluster
    print(f"\n🔗 Step 5: Clustering (DBSCAN eps={args.eps}, min_samples={args.min_samples})...")
    labels = cluster_embeddings(embeddings, results, eps=args.eps, min_samples=args.min_samples)

    # Query demo
    print(f"\n🔍 Step 6: Query demo...")
    if len(results) >= 1:
        sample_desc = results[0]["description"]
        query_similar(collection, sample_desc, sbert, n_results=3)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"   Records processed: {len(results)}")
    print(f"   Embedding dims:    {embeddings.shape[1]}")
    print(f"   Clusters found:    {n_clusters}")
    print(f"   ChromaDB collection: {LOCATION_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
