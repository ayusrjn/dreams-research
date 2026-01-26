### 3.1 Phase 1 — Data Pull & Freezing

## Steps

1. Pull unprocessed records from Cloudflare D1 Database
2. Download images using stored URLs
3. Freeze snapshot for experiments

## Output

```text
data/raw/
  ├── images/
  └── metadata.json
```

This snapshot is the **experiment boundary**.

---

### 3.2 Phase 2 — Feature Extraction (Disentangled)

Each memory is processed into **separate representations**.

---

### A. Image Representation

## Model

- CLIP image encoder (ViT-B/32)

## Purpose

- Capture environmental / scene context

## Explicit Constraint

- Image is **never used** to infer emotion

---

### B. Caption Representation

## Model

- Sentence-BERT (MiniLM) 
### what is to be done 
Strip leading/trailing whitespace

Normalize Unicode

Keep punctuation

Keep casing (MiniLM is case-sensitive)
### what is not to be done 
Strip leading/trailing whitespace

Normalize Unicode

Keep punctuation

Keep casing (MiniLM is case-sensitive)

### Output 
Output

Shape: (384,)

Unit-normalized vector

Stored locally (not Mongo)

## Purpose

- Narrative similarity
- Language structure

---

### C. Emotion Representation

## Source

- Caption text only

## Model

- Pretrained emotion classifier (local, CPU)

### What is to be done 
Input caption text

Tokenize (standard RoBERTa tokenizer)

Forward pass through DistilRoBERTa

Softmax over emotion labels

Output probability distribution

Store valence, arousal, and discrete emotion probabilities

## Representation

```json
{
  "valence": 0.0,
  "arousal": 0.0,
  "joy": 0.0,
  "sadness": 0.0,
  "fear": 0.0,
  "anger": 0.0,
  "neutral": 0.0,
  "disgust": 0.0,
  "surprise": 0.0
}
```

## Key Rule

> Emotion is an estimate of expressed affect, not internal state.

---

### D. Temporal Representation

The goal here is to transform a linear timestamp into features that reveal circadian patterns (time of day) and longitudinal patterns (personal growth/decay).Absolute Timestamp: Preserve ISO-8601 UTC for the master record.Cyclic Time-of-Day (The "Circadian" Feature):Logic: Convert the local hour (0-23) into Sine and Cosine components.Formula: $x = \sin(2\pi \cdot \text{hour}/24)$, $y = \cos(2\pi \cdot \text{hour}/24)$.Purpose: Ensures that 23:00 and 01:00 are mathematically close, allowing the researcher to identify "Late Night" as a singular emotional context.Relative Epoch: * Logic: days_since_start = (current_entry_time - first_entry_time).days.Purpose: Essential for Experiment 2 (Trajectories) to see if emotion in a specific place improves over a semester or a year.

---

### E. Location Representation

To avoid "micro-clusters" caused by GPS drift, we must move from raw coordinates to Categorical Place Identities.Preprocessing (Snap-to-Grid): * Truncate raw Lat/Lon to 4 decimal places before clustering. This provides an immediate $\approx 11\text{m}$ "buffer" against sensor noise.Clustering Algorithm (DBSCAN):Metric: haversine (to account for the Earth's curvature).Epsilon ($\epsilon$): Set to $0.0005$ ($\approx 50\text{m}$). This is the "sweet spot" for distinguishing between "My House" and "The Coffee Shop down the street."Min_Samples: Set to $1$. In this research, even a single entry at a unique location is a valid (though high-variance) data point.Place ID Assignment: * Generate a stable place_id (e.g., P_001).Centroid Calculation: The mean Lat/Lon of all points in the cluster. This becomes the "Anchor Point" for your map visualizations.

## Output

```json
{
  "temporal": {
    "absolute_utc": "2026-01-26T14:30:00Z",
    "relative_day": 42,
    "circadian_coord": {
      "sin_hour": 0.5, 
      "cos_hour": -0.866
    }
  },
  "geospatial": {
    "raw_coord": {"lat": 40.7128, "lon": -74.0060},
    "place_id": "place_03",
    "is_new_cluster": false,
    "place_centroid": {"lat": 40.7129, "lon": -74.0059}
  },
  "research_metadata": {
    "processing_version": "1.0.2",
    "dbscan_eps": 0.0005
  }
}
```

## Key Rule

> Location is categorical context, not a vector.

---

## 4️⃣ Core Data Representation (After Processing)

Each memory becomes:

| Component | Stored Where |
| --- | --- |
| Image embedding | Local vector store |
| Text embedding | Local vector store |
| Emotion (V/A) | Metadata |
| Time | Metadata |
| Place ID | Metadata |

**No early fusion. Ever.**