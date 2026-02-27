import os, re, sys, json, hashlib, logging, urllib.parse
from datetime import datetime

import certifi, requests
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



def fetch_data(db):
    docs = list(db["beehive"].find())
    log.info("Fetched %d documents", len(docs))
    return docs


def normalize_document(doc):
    created_at = doc.get("created_at")
    if isinstance(created_at, dict) and "$date" in created_at:
        created_at = created_at["$date"] if isinstance(created_at["$date"], str) else str(created_at["$date"])
    elif isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    elif created_at is not None:
        created_at = str(created_at)

    geo = doc.get("geo_coord") or {}
    return {
        "user_id": doc.get("user_id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "created_at": created_at,
        "latitude": geo.get("latitude"),
        "longitude": geo.get("longitude"),
    }


def download_image(url, dest_dir, doc_id):
    if not url:
        return None

    fname = f"{hashlib.md5(url.encode()).hexdigest()}.jpg"
    fpath = os.path.join(dest_dir, fname)

    if os.path.exists(fpath):
        log.info("Skipping existing: %s", fname)
        return fname

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(resp.content)
        log.info("Downloaded: %s", fname)
        return fname
    except requests.RequestException as e:
        log.warning("Image download failed for %s: %s", doc_id, e)
        return None


def write_snapshot(records, snapshot_dir):
    path = os.path.join(snapshot_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    log.info("Wrote %d records to %s", len(records), path)


def main():
    uri = os.getenv("MONGO_URI")
    if not uri:
        log.error("MONGO_URI not set")
        sys.exit(1)

    uri = fix_mongo_uri(uri)
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000, tls=True, tlsCAFile=certifi.where())
        try:
            db = client.get_default_database()
        except Exception:
            db = client["dreams"]
    except Exception as e:
        log.error("MongoDB connection failed: %s", e)
        sys.exit(1)

    try:
        raw_docs = fetch_data(db)
    except Exception as e:
        log.error("Fetch failed: %s", e)
        sys.exit(1)

    if not raw_docs:
        log.warning("No documents found")
        return

    records = [normalize_document(doc) for doc in raw_docs]

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = os.path.join(root, "data", "snapshots", f"snapshot_{stamp}")
    img_dir = os.path.join(snap_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for doc, rec in zip(raw_docs, records):
        local = download_image(doc.get("filename"), img_dir, str(doc.get("_id", "unknown")))
        if local:
            rec["image_file"] = local

    write_snapshot(records, snap_dir)
    log.info("Export complete: %s", snap_dir)


if __name__ == "__main__":
    main()
