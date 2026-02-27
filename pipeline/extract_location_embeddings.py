import asyncio
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, RAW_IMAGES_DIR, SENTENCE_BERT_MODEL, LOCATION_COLLECTION_NAME
from db import init_db, get_collection
from location_semantic import reverse_geocode, generate_description, get_gemini_api_key, get_nominatim_user_agent

_blip = None

def get_caption(path):
    global _blip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not _blip:
        _blip = (BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
                 BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device).eval())
    p, m = _blip
    with Image.open(path) as img:
        img_input = p(images=img.convert('RGB'), return_tensors="pt").to(device)
    with torch.no_grad():
        out = m.generate(**img_input, max_new_tokens=25, min_new_tokens=5, num_beams=3)
    return p.decode(out[0], skip_special_tokens=True).strip()

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
    if (lat := rec.get("lat")) is None or (lon := rec.get("lon")) is None: return None
    lat, lon = float(lat), float(lon)
    
    img_path = next((p for p in [RAW_IMAGES_DIR / rec.get("local_image", ""), RAW_IMAGES_DIR / f"{rec.get('id')}.jpg", RAW_IMAGES_DIR / f"{rec.get('id')}.png"] if p.exists()), None)
    if not img_path: return None

    try: caption = get_caption(img_path)
    except Exception: caption = ""

    time.sleep(2)
    try: geocode = await reverse_geocode(lat, lon, user_agent=ua)
    except Exception: geocode = {"display_name": None, "address": None, "raw": None}

    time.sleep(2)
    try: desc = retry(generate_description, lat, lon, geocode, api_key=key, caption=caption)
    except Exception: desc = None

    if not desc: return None
    return {"id": str(rec["id"]), "user_id": rec.get("user_id"), "lat": lat, "lon": lon, "caption": caption, "geocode_display_name": geocode.get("display_name") or "(unknown)", "description": desc}

def main():
    if not RAW_METADATA_PATH.exists(): sys.exit(1)
    with open(RAW_METADATA_PATH) as f: records = json.load(f).get("records", [])

    key, ua = get_gemini_api_key(), get_nominatim_user_agent()
    results = [res for rec in records if (res := asyncio.run(process(rec, key, ua))) and not time.sleep(5)]
    
    if not results: return

    embeddings = SentenceTransformer(SENTENCE_BERT_MODEL).encode([r["description"] for r in results], convert_to_numpy=True, normalize_embeddings=True)

    get_collection(LOCATION_COLLECTION_NAME).upsert(
        ids=[r["id"] for r in results],
        embeddings=embeddings.tolist(),
        documents=[r["description"] for r in results],
        metadatas=[{"user_id": r["user_id"], "lat": r["lat"], "lon": r["lon"], "caption": r["caption"], "geocode_display_name": r["geocode_display_name"]} for r in results]
    )

    conn = init_db()
    for r in results:
        conn.execute("INSERT OR REPLACE INTO location_descriptions (id, user_id, description, geocode_display_name, image_caption) VALUES (?, ?, ?, ?, ?)", (int(r["id"]), r["user_id"], r["description"], r["geocode_display_name"], r["caption"]))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
