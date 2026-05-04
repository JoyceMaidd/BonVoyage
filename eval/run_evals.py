"""
BonVoyage eval harness — tests tools and data pipeline in isolation.
Run: python -m bonvoyage.eval.run_evals

Requires GEMINI_API_KEY and TAVILY_API_KEY in environment or .env
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

from bonvoyage.agent.intent_extractor import extract_trip_intent
from bonvoyage.models.trip_state import TripState, Attraction
from bonvoyage.tools.attraction_search import search_attractions
from bonvoyage.tools.exporter import export_csv

RESULTS_PATH = Path(__file__).parent / "results.jsonl"

TEST_CASES = [
    {
        "id": "paris_student",
        "input": "I am a university student want to do a 4-day solo trip in Paris, I love arts and ballet. One place I must visit is Opera de Paris.",
        "expected_destination": "Paris",
        "expected_duration": 4,
        "expected_interests_contain": ["art", "ballet"],
        "expected_must_visit_contain": ["Opera"],
    },
    {
        "id": "tokyo_retiree",
        "input": "I'm a 68-year-old retiree planning a 7-day trip to Tokyo. I enjoy temples, traditional culture, and local food.",
        "expected_destination": "Tokyo",
        "expected_duration": 7,
        "expected_interests_contain": ["temple", "food"],
        "expected_must_visit_contain": [],
    },
    {
        "id": "barcelona_young",
        "input": "25-year-old traveler, 3 days in Barcelona. Really into architecture and nightlife.",
        "expected_destination": "Barcelona",
        "expected_duration": 3,
        "expected_interests_contain": ["architecture"],
        "expected_must_visit_contain": [],
    },
    {
        "id": "amsterdam_teen",
        "input": "17-year-old on a 5-day trip to Amsterdam with my family. I want to see museums and go cycling.",
        "expected_destination": "Amsterdam",
        "expected_duration": 5,
        "expected_interests_contain": ["museum", "cycling"],
        "expected_must_visit_contain": [],
    },
    {
        "id": "nyc_veteran",
        "input": "US Army veteran, 2 days in New York City. Interested in history and jazz music.",
        "expected_destination": "New York",
        "expected_duration": 2,
        "expected_interests_contain": ["history", "jazz"],
        "expected_must_visit_contain": [],
    },
]


def _check(label: str, passed: bool, details: str = "") -> dict:
    status = "PASS" if passed else "FAIL"
    marker = "✓" if passed else "✗"
    print(f"  {marker} {label}" + (f" — {details}" if details else ""))
    return {"check": label, "status": status, "details": details}


def run_case(tc: dict) -> dict:
    print(f"\n── {tc['id']} ──")
    checks = []
    session_id = f"eval_{tc['id']}"

    # 1. Intent extraction
    intent = extract_trip_intent(tc["input"])
    checks.append(_check(
        "destination extracted",
        tc["expected_destination"].lower() in intent.destination.lower(),
        f"got: '{intent.destination}'",
    ))
    checks.append(_check(
        "duration extracted",
        intent.duration_days == tc["expected_duration"],
        f"got: {intent.duration_days}",
    ))
    for kw in tc["expected_interests_contain"]:
        checks.append(_check(
            f"interest '{kw}' present",
            any(kw.lower() in i.lower() for i in intent.interests),
            f"interests: {intent.interests}",
        ))
    for kw in tc["expected_must_visit_contain"]:
        checks.append(_check(
            f"must_visit '{kw}' present",
            any(kw.lower() in m.lower() for m in intent.must_visit),
            f"must_visit: {intent.must_visit}",
        ))

    # 2. search_attractions schema check
    print(f"  → Searching attractions for {intent.destination}...")
    attractions = search_attractions(
        city=intent.destination or tc["expected_destination"],
        interests=intent.interests or ["sightseeing"],
        must_visit=intent.must_visit,
    )
    checks.append(_check(
        "attractions returned",
        len(attractions) >= 3,
        f"got {len(attractions)} attractions",
    ))
    checks.append(_check(
        "all attractions have name",
        all(bool(a.name) for a in attractions),
    ))
    checks.append(_check(
        "all attractions have description",
        all(bool(a.description) for a in attractions),
    ))
    geocoded = sum(1 for a in attractions if a.lat is not None)
    checks.append(_check(
        "≥50% attractions geocoded",
        geocoded >= len(attractions) / 2,
        f"{geocoded}/{len(attractions)} geocoded",
    ))

    # 3. CSV export check
    state = TripState(
        session_id=session_id,
        destination=intent.destination,
        duration_days=intent.duration_days,
        user_profile=intent.user_profile,
        interests=intent.interests,
        selected_attractions=attractions,
    )
    csv_path_str = export_csv(attractions, "attractions", session_id)
    csv_path = Path(csv_path_str)
    checks.append(_check("CSV file created", csv_path.exists(), str(csv_path)))

    if csv_path.exists():
        header = csv_path.read_text().splitlines()[0]
        required_cols = {"Name", "Address", "Lat", "Lon"}
        checks.append(_check(
            "CSV has required columns",
            all(col in header for col in required_cols),
            f"header: {header}",
        ))

    total = len(checks)
    passed = sum(1 for c in checks if c["status"] == "PASS")
    print(f"  → {passed}/{total} checks passed")

    return {
        "case": tc["id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "total": total,
        "checks": checks,
    }


def main():
    print("BonVoyage Eval Harness")
    print("=" * 40)

    all_results = []
    for tc in TEST_CASES:
        result = run_case(tc)
        all_results.append(result)

    # Write results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # Summary
    total_checks = sum(r["total"] for r in all_results)
    total_passed = sum(r["passed"] for r in all_results)
    print(f"\n{'=' * 40}")
    print(f"TOTAL: {total_passed}/{total_checks} checks passed across {len(TEST_CASES)} cases")
    print(f"Results written to {RESULTS_PATH}")

    if total_passed < total_checks:
        sys.exit(1)


if __name__ == "__main__":
    main()
