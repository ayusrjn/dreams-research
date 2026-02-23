import sys
import os
import asyncio
from google.genai.errors import APIError

sys.path.insert(0, "/home/ayush-ranjan/Documents/dreams-research/pipeline")
from location_semantic.app.services.llm import _get_client
from location_semantic import get_gemini_api_key

api_key = get_gemini_api_key()
client = _get_client(api_key)

try:
    print("Sending 1 request...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, just a test."
    )
    print("Success:", response.text)
except APIError as e:
    print(f"APIError Exception Details: {e.code} / {e.message}")
except Exception as e:
    print(f"General Exception Details: {type(e)} - {e}")

