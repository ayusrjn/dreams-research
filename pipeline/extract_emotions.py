import json
import sys
import unicodedata
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

sys.path.insert(0, str(Path(__file__).parent))
from config import RAW_METADATA_PATH, EMOTION_MODEL, DISCRETE_EMOTION_MODEL, DISCRETE_EMOTIONS
from db import init_db

def main():
    if not RAW_METADATA_PATH.exists():
        sys.exit(1)
        
    with open(RAW_METADATA_PATH) as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL).to(device).eval()
    
    discrete_classifier = pipeline("text-classification", model=DISCRETE_EMOTION_MODEL, top_k=None, device=0 if torch.cuda.is_available() else -1)

    results = []
    
    for record in metadata.get("records", []):
        raw_caption = record.get("caption", "")
        if not raw_caption or raw_caption == "[REDACTED]":
            continue
            
        caption = unicodedata.normalize("NFC", raw_caption).strip()
        if not caption:
            continue
            
        try:
            inputs = tokenizer(caption, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.sigmoid(logits).cpu().numpy()[0]
                valence, arousal = float(preds[0]), float(preds[1]) if logits.shape[-1] >= 2 else 0.5
                
            discrete_output = discrete_classifier(caption)
            emotion_scores = {i["label"]: round(i["score"], 4) for i in (discrete_output[0] if discrete_output else [])}

            r = {"id": record.get("id"), "user_id": record.get("user_id"), "valence": round(valence, 4), "arousal": round(arousal, 4)}
            for e in DISCRETE_EMOTIONS:
                r[e] = emotion_scores.get(e, 0.0)
                
            results.append(r)
        except Exception:
            pass

    if results:
        conn = init_db()
        for r in results:
            conn.execute(
                "INSERT OR REPLACE INTO emotion_scores (id, user_id, valence, arousal, anger, disgust, fear, joy, neutral, sadness, surprise) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r["id"], r["user_id"], r["valence"], r["arousal"], r.get("anger", 0), r.get("disgust", 0), r.get("fear", 0), r.get("joy", 0), r.get("neutral", 0), r.get("sadness", 0), r.get("surprise", 0))
            )
        conn.commit()
        conn.close()

if __name__ == "__main__":
    main()
