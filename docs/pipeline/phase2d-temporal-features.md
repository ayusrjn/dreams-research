# Phase 2D: Temporal Representation

Extracts temporal features from timestamps for circadian and longitudinal analysis.

## Run

```bash
source venv/bin/activate
python pipeline/extract_temporal_features.py
```

## Output

`data/processed/temporal_features.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `id` | Unique record identifier |
| `user_id` | User identifier |
| `absolute_utc` | ISO-8601 UTC timestamp |
| `relative_day` | Days since user's first entry |
| `sin_hour` | sin(2π × hour / 24) - Circadian X coordinate |
| `cos_hour` | cos(2π × hour / 24) - Circadian Y coordinate |

## Circadian Encoding

The sine/cosine encoding ensures temporal continuity:

- 23:00 and 01:00 are mathematically close (wrapped distance)
- Enables smooth gradient-based learning
- Preserves cyclic nature of time-of-day

```python
import math

hour = 14.5  # 2:30 PM
sin_hour = math.sin(2 * math.pi * hour / 24)
cos_hour = math.cos(2 * math.pi * hour / 24)
```
