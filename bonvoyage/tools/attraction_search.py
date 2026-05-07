import json
import os
from tavily import TavilyClient
from google import genai
from pydantic import ValidationError

from bonvoyage.models.trip_state import Attraction
from bonvoyage.tools.geocoder import geocode_address
from bonvoyage.gemini_client import generate_content


def search_attractions(
    city: str,
    interests: list[str],
    must_visit: list[str] | None = None,
) -> list[Attraction]:
    """Search Tavily for attractions, extract structured data via LLM, geocode."""
    interests_str = ", ".join(interests)
    must_visit = must_visit or []

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    query = f"{city} top attractions {interests_str} travel guide things to do"
    results = client.search(query=query, max_results=8, search_depth="basic")

    snippets = "\n\n".join(
        f"Source: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')}"
        for r in results.get("results", [])
    )

    extraction_prompt = f"""You are extracting attraction data from travel search results.

City: {city}
User interests: {interests_str}
Must-visit places: {", ".join(must_visit) if must_visit else "none specified"}

Search results:
{snippets}

Extract up to 8 attractions from these results. Always include the must-visit places if they appear.
For each attraction output a JSON object with exactly these fields:
- name: string (attraction name)
- description: string (1-2 sentence description)
- address: string (street address or neighborhood in {city}, as specific as possible)
- rationale: string (why this matches the user's interests)

Output ONLY a JSON array of objects. No explanation, no markdown fences."""

    raw = generate_content(extraction_prompt)

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    attractions = []
    for item in data:
        try:
            attraction = Attraction(**item)
            lat, lon = geocode_address(attraction.address, city)
            attraction.lat = lat
            attraction.lon = lon
            attractions.append(attraction)
        except (ValidationError, TypeError):
            continue

    return attractions
