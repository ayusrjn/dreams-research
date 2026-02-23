import sys
import os
import asyncio

sys.path.insert(0, "/home/ayush-ranjan/Documents/dreams-research/pipeline")
from location_semantic.app.services.llm import _get_client
from location_semantic import get_gemini_api_key

api_key = get_gemini_api_key()
print(f"Key preview: {api_key[:10]}...")
client = _get_client(api_key)

print(f"Testing Models...")
try:
    models = list(client.models.list())
    for m in models:
        if "flash" in m.name:
            print(m.name)
except Exception as e:
    print(f"List error: {e}")

