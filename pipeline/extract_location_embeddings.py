import asyncio
import json
import sys
import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, RAW_IMAGES_DIR, LOCATION_COLLECTION_NAME
from db import init_db, get_collection
from location_semantic.app.services.geocoder import reverse_geocode

def get_nominatim_user_agent():
    return "dreams-research/1.0 (contact@dreams-research.org)"

def format_geocode_data(geocode_data: dict) -> str:
    """Format geocode data deterministically for CLIP encode"""
    if not geocode_data or not geocode_data.get("raw"):
        return "Unknown location without specific geographic features"
    
    raw = geocode_data["raw"]
    address = geocode_data.get("address", {})
    
    parts = []
    if "city" in address: parts.append(f"City: {address['city']}")
    elif "town" in address: parts.append(f"Town: {address['town']}")
    elif "village" in address: parts.append(f"Village: {address['village']}")
    elif "county" in address: parts.append(f"County: {address['county']}")
    
    if "suburb" in address: parts.append(f"Suburb: {address['suburb']}")
    elif "neighbourhood" in address: parts.append(f"Neighbourhood: {address['neighbourhood']}")
    
    place_type = raw.get("type", "").replace("_", " ")
    place_category = raw.get("category", "").replace("_", " ")
    if place_type: parts.append(f"Type: {place_type}")
    if place_category: parts.append(f"Category: {place_category}")
    
    if "amenity" in address: parts.append(f"Amenity: {address['amenity'].replace('_', ' ')}")
    if "building" in address: parts.append(f"Building: {address['building'].replace('_', ' ')}")
    if "leisure" in address: parts.append(f"Leisure: {address['leisure'].replace('_', ' ')}")
    if "natural" in address: parts.append(f"Natural: {address['natural'].replace('_', ' ')}")
    
    # Remove duplicates
    seen = set()
    unique_parts = [x for x in parts if not (x in seen or seen.add(x))]
    
    return "Location characteristics: " + ", ".join(unique_parts) if unique_parts else "Unknown location without specific geographic features"

async def process_metadata(rec, ua):
    if (lat := rec.get("lat")) is None or (lon := rec.get("lon")) is None: return None
    lat, lon = float(lat), float(lon)
    
    img_path = next((p for p in [RAW_IMAGES_DIR / rec.get("local_image", ""), RAW_IMAGES_DIR / f"{rec.get('id')}.jpg", RAW_IMAGES_DIR / f"{rec.get('id')}.png"] if p.exists()), None)
    if not img_path: return None

    time.sleep(1.5) # respect rate limit to nominatim
    try: geocode = await reverse_geocode(lat, lon, user_agent=ua)
    except Exception: geocode = {"display_name": None, "address": None, "raw": None}

    desc = format_geocode_data(geocode)

    return {
        "id": str(rec["id"]), 
        "user_id": rec.get("user_id"), 
        "lat": lat, 
        "lon": lon, 
        "img_path": img_path,
        "geocode_display_name": geocode.get("display_name") or "(unknown)", 
        "description": desc
    }

def main():
    if not RAW_METADATA_PATH.exists(): sys.exit(1)
    with open(RAW_METADATA_PATH) as f: records = json.load(f).get("records", [])

    ua = get_nominatim_user_agent()
    print(f"Loaded {len(records)} records. Reverse geocoding...")
    results = [res for rec in records if (res := asyncio.run(process_metadata(rec, ua)))]
    
    if not results: 
        print("No valid records found.")
        return

    print(f"Loading CLIP model and computing {len(results)} multi-modal embeddings...")
    clip_model = SentenceTransformer("clip-ViT-B-32")
    
    # Batch encode texts
    texts = [r["description"] for r in results]
    text_embs = clip_model.encode(texts, convert_to_numpy=True)
    
    # Batch encode images
    images = [Image.open(r["img_path"]).convert("RGB") for r in results]
    img_embs = clip_model.encode(images, convert_to_numpy=True)
    
    # Fuse embeddings across modalities (Average fusion)
    multi_embs = (text_embs + img_embs) / 2.0
    
    # Re-normalize to unit length for cosine similarity
    multi_embs = multi_embs / np.linalg.norm(multi_embs, axis=1, keepdims=True)

    print("Upserting to ChromaDB...")
    get_collection(LOCATION_COLLECTION_NAME).upsert(
        ids=[r["id"] for r in results],
        embeddings=multi_embs.tolist(),
        documents=[r["description"] for r in results],
        metadatas=[{"user_id": r["user_id"], "lat": r["lat"], "lon": r["lon"], "geocode_display_name": r["geocode_display_name"]} for r in results]
    )

    print("Saving text representations to SQLite...")
    conn = init_db()
    for r in results:
        conn.execute("INSERT OR REPLACE INTO location_descriptions (id, user_id, description, geocode_display_name, image_caption) VALUES (?, ?, ?, ?, ?)", (int(r["id"]), r["user_id"], r["description"], r["geocode_display_name"], ""))
    conn.commit()
    conn.close()
    
    print("Done!")

if __name__ == "__main__":
    main()
