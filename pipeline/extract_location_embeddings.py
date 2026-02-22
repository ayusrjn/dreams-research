#!/usr/bin/env python3
"""
Phase 2E: Location Embedding Extraction

Processes memory records through:
1. Image captioning (Gemini Vision)
2. Reverse geocoding (Nominatim)
3. Semantic description generation (Gemini text)
4. Description embedding (Sentence-BERT all-MiniLM-L6-v2)
5. Storage in ChromaDB (location_descriptions collection)
6. Storage in SQLite (location_descriptions table)

Stores embeddings in ChromaDB `location_descriptions` collection.
"""

import asyncio
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    RAW_IMAGES_DIR,
    SENTENCE_BERT_MODEL,
    LOCATION_COLLECTION_NAME,
)
from db import init_db, get_collection
from location_semantic import (
    reverse_geocode,
    generate_description,
    get_gemini_api_key,
    get_nominatim_user_agent,
)
from google import genai


# ---------------------------------------------------------------------------
# Image captioning (Gemini Vision)
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
# Retry helper
# ---------------------------------------------------------------------------

def _retry_api(func, *args, max_retries=3, base_wait=25, **kwargs):
    """Call an API function with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = (
                "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                or "503" in err_str or "UNAVAILABLE" in err_str
            )
            if is_rate_limit and attempt < max_retries:
                wait = base_wait * (2 ** attempt)
                print(f"      [WAIT] Rate limited, waiting {wait}s (retry {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """Load the frozen snapshot metadata."""
    if not RAW_METADATA_PATH.exists():
        print(f"[ERROR] Metadata not found: {RAW_METADATA_PATH}")
        print("   Run Phase 1 first: python pipeline/pull_data.py")
        sys.exit(1)

    with open(RAW_METADATA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-record processing
# ---------------------------------------------------------------------------

def resolve_image_path(record: dict) -> Path | None:
    """Find the local image file for a record."""
    local_rel = record.get("local_image")
    if local_rel:
        candidate = RAW_IMAGES_DIR / local_rel
        if candidate.exists():
            return candidate

    record_id = record.get("id")
    for ext in ("jpg", "jpeg", "png", "webp"):
        candidate = RAW_IMAGES_DIR / f"{record_id}.{ext}"
        if candidate.exists():
            return candidate

    return None


async def process_single_record(
    record: dict, api_key: str, user_agent: str
) -> dict | None:
    """Process one metadata record: caption -> geocode -> describe.

    Returns a result dict or None if the record cannot be processed.
    """
    record_id = record.get("id")
    user_id = record.get("user_id")
    lat = record.get("lat")
    lon = record.get("lon")

    if lat is None or lon is None:
        print(f"   [WARN] Record {record_id}: No coordinates (skipped)")
        return None

    lat, lon = float(lat), float(lon)

    image_path = resolve_image_path(record)
    if image_path is None:
        print(f"   [WARN] Record {record_id}: Image not found (skipped)")
        return None

    # 1) Caption
    try:
        caption = _retry_api(get_image_caption, str(image_path), api_key)
    except Exception as e:
        print(f"   [WARN] Record {record_id}: Captioning failed ({e})")
        caption = None

    # Delay: respect Nominatim 1 req/sec + spread Gemini calls
    time.sleep(5)

    # 2) Geocode
    try:
        geocode_data = await reverse_geocode(lat, lon, user_agent=user_agent)
    except Exception as e:
        print(f"   [WARN] Record {record_id}: Geocoding failed ({e})")
        geocode_data = {"display_name": None, "address": None, "raw": None}

    # Delay before next Gemini call
    time.sleep(10)

    # 3) Description
    try:
        description = _retry_api(
            generate_description,
            lat, lon, geocode_data, api_key=api_key, caption=caption,
        )
    except Exception as e:
        print(f"   [WARN] Record {record_id}: Description generation failed ({e})")
        description = None

    if not description:
        print(f"   [WARN] Record {record_id}: No description produced (skipped)")
        return None

    display_name = geocode_data.get("display_name") or "(unknown)"
    print(f"   [OK] Record {record_id}: {display_name}")
    print(f"      Caption:     {caption}")
    print(f"      Description: {description}")

    return {
        "id": str(record_id),
        "user_id": user_id,
        "lat": lat,
        "lon": lon,
        "caption": caption or "",
        "geocode_display_name": display_name,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_embeddings(record_infos: list[dict], embeddings: np.ndarray) -> None:
    """Store embeddings in ChromaDB location_descriptions collection."""
    collection = get_collection(LOCATION_COLLECTION_NAME)

    ids = [r["id"] for r in record_infos]
    documents = [r["description"] for r in record_infos]
    metadatas = [
        {
            "user_id": r["user_id"],
            "lat": r["lat"],
            "lon": r["lon"],
            "caption": r["caption"],
            "geocode_display_name": r["geocode_display_name"],
        }
        for r in record_infos
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    print(f"[INFO] Stored {len(ids)} embeddings in ChromaDB ({LOCATION_COLLECTION_NAME})")
    print(f"   Collection size: {collection.count()}")


def store_descriptions(results: list[dict]) -> None:
    """Store location descriptions in the SQLite location_descriptions table."""
    conn = init_db()

    for r in results:
        conn.execute(
            """INSERT OR REPLACE INTO location_descriptions
               (id, user_id, description, geocode_display_name, image_caption)
               VALUES (?, ?, ?, ?, ?)""",
            (
                int(r["id"]), r["user_id"], r["description"],
                r["geocode_display_name"], r["caption"],
            ),
        )

    conn.commit()
    conn.close()

    print(f"[INFO] Stored {len(results)} records in SQLite (location_descriptions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main execution flow for Phase 2E: Location Embeddings."""
    print("=" * 60)
    print("DREAMS Research - Phase 2E: Location Embedding Extraction")
    print("=" * 60)
    print()

    api_key = get_gemini_api_key()
    user_agent = get_nominatim_user_agent()

    # Step 1: Load metadata
    print("[INFO] Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")

    # Step 2: Process records (caption -> geocode -> describe)
    records = metadata.get("records", [])
    print(f"\n[INFO] Step 2: Processing {len(records)} records (caption -> geocode -> describe)...")
    results = []
    for i, rec in enumerate(records):
        print(f"\n   --- Record {i+1}/{len(records)} ---")
        result = asyncio.run(
            process_single_record(rec, api_key, user_agent)
        )
        if result:
            results.append(result)

        # Delay between records to stay within Gemini rate limits
        if i < len(records) - 1:
            print("   [WAIT] Pausing 10s before next record...")
            time.sleep(10)

    if not results:
        print("\n[WARN] No descriptions extracted. Check your images and API key.")
        return

    print(f"\n   Processed: {len(results)}/{len(records)} records")

    # Step 3: Embed descriptions
    print(f"\n[INFO] Step 3: Embedding descriptions ({SENTENCE_BERT_MODEL})...")
    model = SentenceTransformer(SENTENCE_BERT_MODEL)
    descriptions = [r["description"] for r in results]
    embeddings = model.encode(
        descriptions,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"   Shape: {embeddings.shape}")

    # Step 4: Store in ChromaDB
    print(f"\n[INFO] Step 4: Storing in ChromaDB...")
    store_embeddings(results, embeddings)

    # Step 5: Store in SQLite
    print(f"\n[INFO] Step 5: Storing in SQLite...")
    store_descriptions(results)

    # Summary
    print("\n" + "=" * 60)
    print("[OK] Phase 2E Complete!")
    print("=" * 60)
    print(f"   [INFO] Embeddings: {embeddings.shape[0]} locations")
    print(f"   [INFO] Dimensions: {embeddings.shape[1]}")
    print(f"   [INFO] Collection: {LOCATION_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
