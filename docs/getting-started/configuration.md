# Configuration

## Environment Variables

The pipeline requires configuration via environment variables for database access.

### Setup

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` with your credentials:

```bash
# Cloudflare D1 Database
D1_DATABASE_ID=your_database_id
D1_API_TOKEN=your_api_token
CLOUDFLARE_ACCOUNT_ID=your_account_id
```

### D1 Database Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Primary key |
| `user_id` | string | User UUID |
| `caption` | string | Memory caption |
| `timestamp` | datetime | When memory was captured |
| `lat` | float | Latitude |
| `lon` | float | Longitude |
| `image_url` | string | Cloudinary URL |
| `processed` | int | 0 = unprocessed |
| `processing_version` | string | Pipeline version |
| `created_at` | datetime | DB insert time |

## Pipeline Configuration

Pipeline parameters are defined in `pipeline/config.py`:

- **CLIP Model**: ViT-B/32
- **Sentence-BERT Model**: all-MiniLM-L6-v2
- **DBSCAN ε**: ~50 meters (7.85×10⁻⁶ radians)
