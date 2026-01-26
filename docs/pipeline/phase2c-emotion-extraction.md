# Phase 2C: Emotion Extraction

Extracts emotional features from captions using pretrained models.

## Models Used

| Model | Output |
|-------|--------|
| Mavdol/NPC-Valence-Arousal-Prediction | Valence/Arousal dimensions |
| j-hartmann/emotion-english-distilroberta-base | Discrete emotion probabilities |

## Run

```bash
source venv/bin/activate
python pipeline/extract_emotions.py
```

## Output

`data/processed/emotion_scores.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique record identifier |
| `user_id` | User identifier |
| `valence` | Pleasant (1) ↔ Unpleasant (0) |
| `arousal` | High energy (1) ↔ Low energy (0) |
| `joy` | Probability of joy |
| `sadness` | Probability of sadness |
| `fear` | Probability of fear |
| `anger` | Probability of anger |
| `neutral` | Probability of neutral |
| `disgust` | Probability of disgust |
| `surprise` | Probability of surprise |

!!! note "Interpretation"
    Emotion scores are an estimate of **expressed affect**, not internal state.
