import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache
def get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in environment or .env file")
    return key


@lru_cache
def get_nominatim_user_agent() -> str:
    return os.getenv("NOMINATIM_USER_AGENT", "location-semantic-api/1.0")
