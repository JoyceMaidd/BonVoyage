import json
from pydantic import BaseModel
from bonvoyage.gemini_client import generate_content


class ExtractedIntent(BaseModel):
    destination: str = ""
    duration_days: int = 0
    user_profile: str = ""
    interests: list[str] = []
    must_visit: list[str] = []
    missing_fields: list[str] = []


def extract_trip_intent(user_input: str) -> ExtractedIntent:
    """Parse natural language trip request into structured fields."""
    prompt = f"""Extract travel planning information from this user message.

User message: "{user_input}"

Extract and return a JSON object with exactly these fields:
- destination: string (destination name, empty string if not mentioned); The field is required.
- duration_days: integer (number of days, 0 if not mentioned); The field is required.
- user_profile: string (traveler description e.g. "university student", "68-year-old retiree", empty if not mentioned)
- interests: array of strings (hobbies, activities, interests mentioned)
- must_visit: array of strings (specific places the user says they must visit or want to see)
- missing_fields: array of strings — list only fields that are REQUIRED but missing from the user input (e.g. ["destination", "duration_days"] if both are missing, or [] if all required fields are present)

Examples of interests: "modern art", "ballet", "street food", "architecture", "nightlife", "hiking"
Examples of must_visit: "Eiffel Tower", "Opera de Paris", "Sagrada Familia"

Output ONLY the JSON object. No explanation, no markdown fences."""

    raw = generate_content(prompt)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        print(f"Extracted intent: {data}")
        return ExtractedIntent(**data)
    except (json.JSONDecodeError, Exception):
        return ExtractedIntent(missing_fields=["destination", "duration_days"])


def build_missing_fields_prompt(intent: ExtractedIntent) -> str:
    """Build one consolidated follow-up question for missing fields."""
    missing = intent.missing_fields
    if not missing:
        return ""

    need_dest = "destination" in missing
    need_dur = "duration_days" in missing

    if need_dest and need_dur:
        return "Which place are you visiting, and for how many days?"
    if need_dest:
        return "Which place are you planning to visit?"
    if need_dur:
        destination = intent.destination or "your destination"
        return f"How many days are you planning to stay in {destination}?"
    return ""