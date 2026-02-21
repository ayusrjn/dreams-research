import sys
from pathlib import Path

# The inner `app` package uses absolute-style imports like
# `from app.config import ...`, which require `location_semantic/`
# itself to be on sys.path. Ensure that before importing.
_pkg_dir = str(Path(__file__).resolve().parent)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from app import (  # noqa: E402
    reverse_geocode,
    generate_description,
    get_gemini_api_key,
    get_nominatim_user_agent,
)

__all__ = [
    "reverse_geocode",
    "generate_description",
    "get_gemini_api_key",
    "get_nominatim_user_agent",
]
