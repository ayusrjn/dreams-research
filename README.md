# DREAMS: 

**DREAMS** is a computational research pipeline designed to quantitatively validate the existence of "Stable Emotional Fingerprints" in human memory. By disentangling the multimodal dimensions of memory streams—visual, narrative, spatial, and temporal—this project seeks to determine if specific physical locations induce statistically consistent emotional states over time.

---

##  Research Hypothesis

The core premise of this research is that physical and semantic locations possess a **Stable Emotional Fingerprint**. We hypothesize that when a user visits the same place repeatedly, their emotional state converges to a consistent, statistically stable pattern, independent of transient mood fluctuations.

To test this, we define the analysis unit at the $(u, p) = (user\_id, place\_id)$ level to preserve individual narrative contexts.

---

##  Mathematical Framework

We model the emotional state of a memory at time $t$ as a low-dimensional continuous vector:

$$C_{t} = [valence, arousal]$$

where $valence \in [0, 1]$ (Unpleasant $\to$ Pleasant) and $arousal \in [0, 1]$ (Calm $\to$ Excited).

To quantify the stability of a location's fingerprint, we compute the following metrics for each user-place pair $(u, p)$:

### 1. Mean Emotional State (The "Center")
The expected emotional baseline for a specific location:
$$\mu_{u,p} = \mathbb{E}[C_{t}]$$

### 2. Emotional Variability
We measure the dispersion of emotional states using the covariance matrix:
$$\Sigma_{u,p} = Cov(C_{t})$$

### 3. Stability (Variance Ellipse)
The volatility of the location is proportional to the area of the variance ellipse defined by $\Sigma_{u,p}$:
$$A_{u,p} \propto \sqrt{|\Sigma_{u,p}|}$$
*   **Small Area**: High stability (Strong Fingerprint).
*   **Large Area**: Low stability (Volatile Context).

### 4. Emotion Entropy
To assess the consistency of discrete emotion types (e.g., Joy vs. Fear), we calculate the entropy of the probability distribution:
$$H_{u,p} = -\sum_{k} \overline{P}_{u,p}^{k} \log \overline{P}_{u,p}^{k}$$

---

##  Pipeline Architecture

The pipeline executes in three phases to transform raw logs into research-ready vectors.

### Phase 1: Acquisition & Freezing
*   **Objective**: Establish an immutable "Snapshot" of the raw data.
*   **Process**: Pulls multimodal logs (images, captions, metadata) from the Cloudflare D1 database and freezes them to ensure reproducibility.

### Phase 2: Feature Extraction
We extract disentangled representations using state-of-the-art models:
*   **Visual**: CLIP (ViT-B/32) for scene semantics.
*   **Semantic**: Sentence-BERT (`all-MiniLM-L6-v2`) for narrative structure.
*   **Spatial**: **DBSCAN Clustering** ($\epsilon \approx 50m$) converts raw GPS coordinates into categorical Place IDs ($p$), enabling the $(u, p)$ analysis.
*   **Temporal**: Cyclic encoding ($\sin/\cos$) of time-of-day.

### Phase 3: Grand Fusion
*   **Objective**: Synthesis.
*   **Process**: Aligns all scalar features into a `master_manifest.parquet` and synchronizes high-dimensional arrays (`.npy`) for longitudinal analysis.

---

## Research Outcomes

Based on the stability metrics defined above, we aim to categorize locations into three distinct types relative to a user's baseline:

1.  **Emotional Safe Space**: High Valence, Low Arousal, **Low Variance** ($A_{u,p} \to 0$).
2.  **Chronic Stressor**: Low Valence, High Arousal, **Low Variance** (Consistently negative).
3.  **Emotionally Volatile**: High Variance ($A_{u,p} \to \infty$), indicating the location does not exert a strong emotional anchor.

---

##  Future Directions

This framework lays the groundwork for distinguishing between **Scene-Driven** vs. **Place-Driven** stability. Future experiments will test whether emotional consistency is driven by visual similarity (e.g., "I feel calm when I see trees") or contextual identity (e.g., "I feel calm because I am at Home," regardless of the visual view).

---

##  Getting Started

### Prerequisites
*   Python 3.10+
*   CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/ayusrjn/dreams-research.git
cd dreams-research
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Reproduction

```bash
# 1. Pull Data (→ metadata.json + SQLite memories table)
python pipeline/pull_data.py

# 2. Extract Features
python pipeline/extract_image_embeddings.py    # → ChromaDB: image_embeddings
python pipeline/extract_caption_embeddings.py  # → ChromaDB: caption_embeddings
python pipeline/extract_emotions.py            # → SQLite: emotion_scores
python pipeline/extract_temporal_features.py   # → SQLite: temporal_features
python pipeline/extract_location_clusters.py   # → SQLite: place_assignments

# 3. Verify (master_manifest is a SQL VIEW — always in sync)
python pipeline/create_master_manifest.py

# Optional: export to Parquet for backward compatibility
python pipeline/create_master_manifest.py --export
```

---

##  Data Schema

### SQLite (`data/processed/dreams.db`)

| Table / View | Description |
| :--- | :--- |
| `memories` | Raw record metadata (user, caption, timestamp, GPS, image path). |
| `emotion_scores` | Valence, arousal, and 7 discrete emotion probabilities ($C_t$). |
| `temporal_features` | Circadian sin/cos encoding + relative epoch (days since first entry). |
| `place_assignments` | DBSCAN place IDs, snapped coordinates, cluster centroids. |
| `master_manifest` | **VIEW** joining all tables — the unified research dataset. |

### ChromaDB (`data/processed/chroma_db/`)

| Collection | Shape | Description |
| :--- | :--- | :--- |
| `image_embeddings` | $(N, 512)$ | CLIP Visual Embeddings. |
| `caption_embeddings` | $(N, 384)$ | S-BERT Narrative Embeddings. |
| `location_descriptions` | $(N, 384)$ | S-BERT Location Semantic Embeddings. |

---

