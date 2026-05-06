import json
import os
from tavily import TavilyClient
from google import genai
from pydantic import ValidationError

from bonvoyage.models.trip_state import Discount
from bonvoyage.gemini_client import generate_content


def lookup_discounts(city: str, user_profile: str) -> list[Discount]:
    """Search for discounts relevant to the user profile. Best-effort results."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    query = f"{city} discounts {user_profile} museums attractions transit student youth senior"
    results = client.search(query=query, max_results=5, search_depth="basic")

    snippets = "\n\n".join(
        f"Source: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')}"
        for r in results.get("results", [])
    )

    extraction_prompt = f"""You are identifying discount opportunities for a traveler.

City: {city}
Traveler profile: {user_profile}

Search results:
{snippets}

Extract discounts that are plausibly relevant to this traveler profile.
Only include discounts you can reasonably infer from the search results — do not fabricate.
For each discount output a JSON object with exactly these fields:
- name: string (discount name or type, e.g. "Museum Pass", "Youth Transit Discount")
- description: string (what the discount offers and where to get it)
- eligibility: string (who qualifies — be specific about age/status requirements)

Output ONLY a JSON array of objects. If no relevant discounts found, output an empty array [].
No explanation, no markdown fences."""

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

    discounts = []
    for item in data:
        try:
            discounts.append(Discount(**item))
        except (ValidationError, TypeError):
            continue

    return discounts
