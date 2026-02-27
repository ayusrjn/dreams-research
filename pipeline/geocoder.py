import httpx
from typing import Any

NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"


async def reverse_geocode(
    lat: float, lon: float, user_agent: str
) -> dict[str, Any]:
    """Reverse-geocode coordinates using OpenStreetMap Nominatim.

    Returns a dict with keys: display_name, address, raw.
    If the location cannot be resolved, display_name and address may be None.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
        "extratags": 1,
        "namedetails": 1,
    }
    headers = {"User-Agent": user_agent}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{NOMINATIM_BASE_URL}/reverse", params=params, headers=headers
        )

        if response.status_code == 404 or (
            response.status_code == 200
            and response.json().get("error")
        ):
            return {
                "display_name": None,
                "address": None,
                "raw": None,
            }

        response.raise_for_status()
        data = response.json()

        return {
            "display_name": data.get("display_name"),
            "address": data.get("address", {}),
            "raw": data,
        }
