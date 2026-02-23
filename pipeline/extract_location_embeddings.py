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
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# ---------------------------------------------------------------------------
# Global BLIP Model (Loaded on-demand)
# ---------------------------------------------------------------------------
_blip_processor = None
_blip_model = None
_blip_device = "cpu"

def load_blip_model():
    """Load the BLIP model locally for free image captioning."""
    global _blip_processor, _blip_model, _blip_device
    if _blip_processor is None:
        print("[INFO] Loading local BLIP image captioning model (Salesforce/blip-image-captioning-base)...")
        _blip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(_blip_device)
        _blip_model.eval()
    return _blip_processor, _blip_model


# ---------------------------------------------------------------------------
# Image captioning (Local BLIP)
# ---------------------------------------------------------------------------

def get_image_caption(image_path: str) -> str:
    """Generate a short caption for an image using a local BLIP model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    processor, model = load_blip_model()
    
    with Image.open(image_path) as img:
        # Convert to RGB if needed (e.g., RGBA or grayscale)
        raw_image = img.convert('RGB')
        
    # Process image
    inputs = processor(images=raw_image, return_tensors="pt").to(_blip_device)
    
    # Generate caption (we want a short summary similar to what Gemini gave)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=25,
            min_new_tokens=5,
            num_beams=3
        )
        
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_api(func, *args, max_retries=3, base_wait=25, **kwargs):
    """Call an API function with exponential backoff on rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            
            # Check for hard quota exhaustion first
            if "quota" in err_str and "exceeded" in err_str:
                print(f"      [ERROR] Hard quota exceeded on API key. Aborting retries.")
                raise e
            
            # Otherwise, check for normal rate limits (per minute/day constraints)
            is_rate_limit = (
                "429" in err_str or "resource_exhausted" in err_str
                or "503" in err_str or "unavailable" in err_str
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

    # 1) Caption (Local, no retry needed)
    try:
        caption = get_image_caption(str(image_path))
    except Exception as e:
        print(f"   [WARN] Record {record_id}: Captioning failed ({e})")
        caption = None

    # Delay: respect Nominatim 1 req/sec
    time.sleep(2)

    # 2) Geocode
    try:
        geocode_data = await reverse_geocode(lat, lon, user_agent=user_agent)
    except Exception as e:
        print(f"   [WARN] Record {record_id}: Geocoding failed ({e})")
        geocode_data = {"display_name": None, "address": None, "raw": None}

    # Delay before Gemini call
    time.sleep(2)

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
            print("   [WAIT] Pausing 5s before next record...")
            time.sleep(5)

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
