import json
import sys
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, RAW_IMAGES_DIR, CLIP_MODEL, IMAGE_COLLECTION_NAME
from db import get_collection

def main():
    if not RAW_METADATA_PATH.exists():
        sys.exit(1)
        
    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    embeddings, record_infos = [], []
    
    for record in metadata.get("records", []):
        if not (local_image := record.get("local_image")):
            continue
            
        image_path = RAW_IMAGES_DIR / local_image
        if not image_path.exists():
            found = list(RAW_IMAGES_DIR.rglob(Path(local_image).name))
            if not found:
                continue
            image_path = found[0]
            
        try:
            with Image.open(image_path) as img:
                image_input = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image_input).cpu().numpy().flatten()
            embeddings.append(emb)
            record_infos.append({"id": str(record.get("id")), "user_id": record.get("user_id"), "local_image": local_image})
        except Exception:
            pass

    if embeddings:
        get_collection(IMAGE_COLLECTION_NAME).upsert(
            ids=[r["id"] for r in record_infos],
            embeddings=np.stack(embeddings).tolist(),
            metadatas=[{"user_id": r["user_id"], "local_image": r["local_image"]} for r in record_infos]
        )

if __name__ == "__main__":
    main()
