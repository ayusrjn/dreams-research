from typing import Any
from google import genai

_client: genai.Client | None = None


def _get_client(api_key: str) -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


def _build_prompt(
    lat: float, lon: float, geocode_data: dict[str, Any], caption: str | None = None
) -> str:
    """Build a prompt for the LLM from geocode results and an optional image caption."""
    display_name = geocode_data.get("display_name") or "Unknown location"
    address = geocode_data.get("address") or {}
    raw = geocode_data.get("raw") or {}

    address_parts = []
    for key in [
        "amenity", "building", "road", "neighbourhood", "suburb",
        "city", "town", "village", "state", "country",
    ]:
        if key in address:
            address_parts.append(f"{key}: {address[key]}")

    extra_tags = raw.get("extratags") or {}
    name_details = raw.get("namedetails") or {}
    place_type = raw.get("type", "")
    place_category = raw.get("category", "")

    extras = []
    if place_type:
        extras.append(f"Place type: {place_type}")
    if place_category:
        extras.append(f"Category: {place_category}")
    for k, v in extra_tags.items():
        extras.append(f"{k}: {v}")
    for k, v in name_details.items():
        if k != "name":
            extras.append(f"Name ({k}): {v}")

    caption_section = ""
    if caption:
        caption_section = f"""\nImage caption (from CLIP model):
  {caption}

Use this visual description to enrich your response — combine what is seen in the 
image with the geographic data to paint a complete picture of where the user is."""

    prompt = f"""You are a knowledgeable geographic assistant. Given the following location data
and an optional image caption from a CLIP vision model, write a vivid and informative 
3-5 sentence semantic description of this place. Describe what kind of area it is, 
what a visitor might experience, and any notable geographic or cultural context. 
Be specific and avoid generic filler.

Coordinates: {lat}, {lon}
Display name: {display_name}
Address components:
{chr(10).join(address_parts) if address_parts else "  (none available)"}
Additional details:
{chr(10).join(extras) if extras else "  (none available)"}{caption_section}

Respond with ONLY the description, no preamble or labels."""

    return prompt


def generate_description(
    lat: float,
    lon: float,
    geocode_data: dict[str, Any],
    api_key: str,
    caption: str | None = None,
) -> str:
    """Generate a semantic description of a location using Google Gemini."""
    client = _get_client(api_key)
    prompt = _build_prompt(lat, lon, geocode_data, caption=caption)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    return response.text.strip()
