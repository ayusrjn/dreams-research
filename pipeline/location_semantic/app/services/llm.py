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


    prompt = f"""
You are a structured geographic classifier.

Task:
Convert the following structured metadata into a compact,
standardized semantic description.

Rules:
- 1 sentence only.
- Maximum 25 words.
- No storytelling.
- No sensory language.
- No prestige framing.
- No cultural commentary.
- No geographic landmarks unless explicitly a landmark.
- Use consistent phrasing across similar place types.

Output format:
"<primary_type> in a <environment_type> area. Key attributes: <comma-separated attributes>."

If attributes are missing, omit the attributes section.

Metadata:
Type: {place_type}
Category: {place_category}
Address:
{chr(10).join(address_parts) if address_parts else "None"}
Extra tags:
{chr(10).join([f"{k}: {v}" for k, v in extra_tags.items()]) if extra_tags else "None"}
Caption summary:
{caption if caption else "None"}

Return ONLY the sentence.
"""
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
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()
