#!/usr/bin/env python3
"""
Phase 2C: Emotion Extraction

Extracts both:
1. Valence/arousal scores (dimensional)
2. Discrete emotion probabilities (joy, sadness, fear, anger, neutral, etc.)

Stores results in the SQLite `emotion_scores` table.
"""

import json
import sys
import unicodedata
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_METADATA_PATH,
    EMOTION_MODEL,
    DISCRETE_EMOTION_MODEL,
    DISCRETE_EMOTIONS,
)
from db import init_db


def load_metadata() -> dict:
    if not RAW_METADATA_PATH.exists():
        print(f"[ERROR] Metadata not found: {RAW_METADATA_PATH}")
        print("   Run Phase 1 first: python pipeline/pull_data.py")
        sys.exit(1)
    
    with open(RAW_METADATA_PATH) as f:
        return json.load(f)


def load_valence_arousal_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading valence/arousal model: {EMOTION_MODEL} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL)
    model.to(device)
    model.eval()
    
    return tokenizer, model, device


def load_discrete_emotion_model():
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"[INFO] Loading discrete emotion model: {DISCRETE_EMOTION_MODEL}")
    
    classifier = pipeline(
        "text-classification",
        model=DISCRETE_EMOTION_MODEL,
        top_k=None,
        device=device_id
    )
    return classifier


def preprocess_caption(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    return text


def extract_emotions(metadata: dict, va_tokenizer, va_model, va_device, discrete_classifier) -> list[dict]:
    records = metadata.get("records", [])
    results = []
    
    print(f"[INFO] Processing {len(records)} records...")
    
    for record in records:
        record_id = record.get("id")
        user_id = record.get("user_id")
        raw_caption = record.get("caption", "")
        
        if not raw_caption or raw_caption == "[REDACTED]":
            print(f"   [WARN] Record {record_id}: No valid caption (skipped)")
            continue
        
        caption = preprocess_caption(raw_caption)
        if not caption:
            print(f"   [WARN] Record {record_id}: Empty after preprocessing (skipped)")
            continue
        
        try:
            inputs = va_tokenizer(
                caption,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(va_device)
            
            with torch.no_grad():
                outputs = va_model(**inputs)
                logits = outputs.logits
                
                if logits.shape[-1] >= 2:
                    predictions = torch.sigmoid(logits).cpu().numpy()[0]
                    valence = float(predictions[0])
                    arousal = float(predictions[1])
                else:
                    predictions = torch.sigmoid(logits).cpu().numpy()[0]
                    valence = float(predictions[0])
                    arousal = 0.5
            
            discrete_output = discrete_classifier(caption)
            discrete_results = discrete_output[0] if discrete_output else []
            emotion_scores = {item["label"]: round(item["score"], 4) for item in discrete_results}
            
            result = {
                "id": record_id,
                "user_id": user_id,
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
            }
            for emotion in DISCRETE_EMOTIONS:
                result[emotion] = emotion_scores.get(emotion, 0.0)
            
            results.append(result)
            
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            print(f"   [OK] Record {record_id}: V={valence:.2f} A={arousal:.2f} | {top_emotion}={emotion_scores[top_emotion]:.2f}")
            
        except Exception as e:
            print(f"   [ERROR] Record {record_id}: Failed - {e}")
            continue
    
    return results


def store_emotions(results: list[dict]) -> None:
    """Store emotion scores in the SQLite emotion_scores table."""
    conn = init_db()
    
    for r in results:
        conn.execute(
            """INSERT OR REPLACE INTO emotion_scores
               (id, user_id, valence, arousal,
                anger, disgust, fear, joy, neutral, sadness, surprise)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["id"], r["user_id"], r["valence"], r["arousal"],
                r.get("anger", 0), r.get("disgust", 0), r.get("fear", 0),
                r.get("joy", 0), r.get("neutral", 0), r.get("sadness", 0),
                r.get("surprise", 0),
            ),
        )
    
    conn.commit()
    conn.close()
    
    print(f"[INFO] Stored {len(results)} emotion records in SQLite (emotion_scores)")


def main():
    print("=" * 60)
    print("DREAMS Research - Phase 2C: Emotion Extraction")
    print("=" * 60)
    print()
    
    print("[INFO] Step 1: Loading metadata...")
    metadata = load_metadata()
    print(f"   Snapshot: {metadata.get('snapshot_id')}")
    print(f"   Records: {metadata.get('record_count')}")
    
    print("\n[INFO] Step 2: Loading emotion models...")
    va_tokenizer, va_model, va_device = load_valence_arousal_model()
    discrete_classifier = load_discrete_emotion_model()
    
    print("\n[INFO] Step 3: Extracting emotions...")
    results = extract_emotions(metadata, va_tokenizer, va_model, va_device, discrete_classifier)
    
    if not results:
        print("\n[WARN] No emotions extracted. Check your captions.")
        return
    
    print("\n[INFO] Step 4: Storing in SQLite...")
    store_emotions(results)
    
    print("\n" + "=" * 60)
    print("[OK] Phase 2C Complete!")
    print("=" * 60)
    print(f"   [INFO] Processed: {len(results)} captions")


if __name__ == "__main__":
    main()
