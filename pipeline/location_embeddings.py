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
from config import RAW_IMAGES_DIR, RAW_METADATA_PATH, SENTENCE_BERT_MODEL, LOCATION_COLLECTION_NAME
from db import get_collection
from location_semantic import reverse_geocode, generate_description, get_gemini_api_key, get_nominatim_user_agent

def get_caption(path, key):
    img = Path(path)
    if not img.exists(): raise FileNotFoundError
    b64 = base64.standard_b64encode(img.read_bytes()).decode("utf-8")
    resp = genai.Client(api_key=key).models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"inline_data": {"mime_type": mimetypes.guess_type(path)[0] or "image/jpeg", "data": b64}}, {"text": "Describe the environment shown in this image in ONE concise sentence (max 25 words). Focus only on type of place, setting, and functional purpose. Be factual."}]}]
    )
    return resp.text.strip()

async def process(rec, key, ua):
    if (lat := rec.get("lat")) is None or (lon := rec.get("lon")) is None: return None
    lat, lon = float(lat), float(lon)

    img_path = next((p for p in [RAW_IMAGES_DIR / rec.get("local_image", ""), RAW_IMAGES_DIR / f"{rec.get('id')}.jpg", RAW_IMAGES_DIR / f"{rec.get('id')}.png"] if p.exists()), None)
    if not img_path: return None

    try: caption = get_caption(str(img_path), key)
    except Exception: caption = ""

    try: geo = await reverse_geocode(lat, lon, user_agent=ua)
    except Exception: geo = {"display_name": None, "address": None, "raw": None}

    try: desc = generate_description(lat, lon, geo, api_key=key, caption=caption)
    except Exception: desc = None

    if not desc: return None
    return {"id": str(rec["id"]), "lat": lat, "lon": lon, "caption": caption, "geocode_display_name": geo.get("display_name") or "(unknown)", "description": desc}

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--batch", action="store_true")
    args.add_argument("--query", type=str)
    args.add_argument("--image", type=str)
    args.add_argument("--lat", type=float)
    args.add_argument("--lon", type=float)
    args.add_argument("--eps", type=float, default=0.3)
    args.add_argument("--min-samples", type=int, default=2)
    args = args.parse_args()

    if args.query:
        sbert, coll = SentenceTransformer(SENTENCE_BERT_MODEL), get_collection(LOCATION_COLLECTION_NAME)
        if coll.count() == 0: sys.exit(1)
        res = coll.query(query_embeddings=sbert.encode([args.query], normalize_embeddings=True).tolist(), n_results=min(3, coll.count()), include=["documents", "metadatas", "distances"])
        for d, m in zip(res["documents"][0], res["metadatas"][0]): print(f"{m.get('geocode_display_name')}: {d}")
        return

    key, ua = get_gemini_api_key(), get_nominatim_user_agent()

    if args.image and args.lat is not None and args.lon is not None:
        rid = hashlib.md5(f"{args.lat}:{args.lon}:{time.time()}".encode()).hexdigest()[:12]
        cap = get_caption(args.image, key)
        geo = asyncio.run(reverse_geocode(args.lat, args.lon, user_agent=ua))
        desc = generate_description(args.lat, args.lon, geo, api_key=key, caption=cap)
        get_collection(LOCATION_COLLECTION_NAME).upsert(ids=[rid], embeddings=SentenceTransformer(SENTENCE_BERT_MODEL).encode([desc], normalize_embeddings=True).tolist(), metadatas=[{"lat": args.lat, "lon": args.lon, "caption": cap, "geocode_display_name": geo.get("display_name")}])
        return

    if not RAW_METADATA_PATH.exists(): sys.exit(1)
    with open(RAW_METADATA_PATH) as f: records = json.load(f).get("records", [])

    results = [res for rec in records if (res := asyncio.run(process(rec, key, ua)))]
    if not results: sys.exit(1)

    sbert = SentenceTransformer(SENTENCE_BERT_MODEL)
    emb = sbert.encode([r["description"] for r in results], convert_to_numpy=True, normalize_embeddings=True)

    get_collection(LOCATION_COLLECTION_NAME).upsert(ids=[r["id"] for r in results], embeddings=emb.tolist(), documents=[r["description"] for r in results], metadatas=[{"lat": r["lat"], "lon": r["lon"], "caption": r["caption"], "geocode_display_name": r["geocode_display_name"]} for r in results])

    dist = np.clip(1.0 - np.dot(emb, emb.T), 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="precomputed").fit_predict(dist)

if __name__ == "__main__":
    main()
