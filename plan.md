### 3.1 Phase 1 — Data Pull & Freezing

**Steps**

1. Pull unprocessed records from Cloudflare D1 Database
2. Download images using stored URLs
3. Freeze snapshot for experiments

**Output**

```
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

**Model**

- CLIP image encoder (ViT-B/32)

**Purpose**

- Capture environmental / scene context

**Explicit constraint**

- Image is **never used** to infer emotion

---

### B. Caption Representation

**Model**

- Sentence-BERT (MiniLM)

**Purpose**

- Narrative similarity
- Language structure

---

### C. Emotion Representation

**Source**

- Caption text only

**Model**

- Pretrained emotion classifier (local, CPU)

**Representation**

```json
{
"valence": float,
"arousal": float,
"confidence": float
}

```

**Key rule**

> Emotion is an estimate of expressed affect, not internal state.
> 

---

### D. Temporal Representation

- Absolute timestamp
- Relative time (days since first entry)
- Used for trajectory analysis

---

### E. Location Representation

- Raw lat/lon preserved
- Offline clustering (DBSCAN / radius)

**Output**

```json
{
"place_id":"place_03",
"centroid":{"lat": ...,"lon": ...}
}

```

**Key rule**

> Location is categorical context, not a vector.
> 

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