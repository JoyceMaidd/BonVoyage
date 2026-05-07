import json
import os
from tavily import TavilyClient
from google import genai
from pydantic import ValidationError

from bonvoyage.models.trip_state import Hostel
from bonvoyage.gemini_client import generate_content


def find_hostels(city: str) -> list[Hostel]:
    """Search Tavily for hostels, extract structured data via LLM."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    query = f"best hostels in {city} solo traveler budget accommodation"
    results = client.search(query=query, max_results=6, search_depth="basic")

    snippets = "\n\n".join(
        f"Source: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')}"
        for r in results.get("results", [])
    )

    extraction_prompt = f"""You are extracting hostel data from travel search results.

City: {city}

Search results:
{snippets}

Extract up to 5 hostels from these results.
For each hostel output a JSON object with exactly these fields:
- name: string (hostel name)
- description: string (1-2 sentence description, include price range if available)
- address: string (street address or neighborhood in {city})

Output ONLY a JSON array of objects. No explanation, no markdown fences."""

    raw = generate_content(extraction_prompt)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    hostels = []
    for item in data:
        try:
            hostels.append(Hostel(**item))
        except (ValidationError, TypeError):
            continue

    return hostels
