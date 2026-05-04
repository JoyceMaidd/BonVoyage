from bonvoyage.models.trip_state import TripPhase

SEARCH_ATTRACTIONS = {
    "name": "search_attractions",
    "description": "Search for attractions in the destination city that match the user's interests.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The destination city"},
            "interests": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of user interests (e.g. 'modern art', 'ballet')",
            },
            "must_visit": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Places the user has specifically said they must visit",
            },
        },
        "required": ["city", "interests"],
    },
}

FIND_HOSTELS = {
    "name": "find_hostels",
    "description": "Find hostels in the destination city suitable for solo travelers.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The destination city"},
        },
        "required": ["city"],
    },
}

LOOKUP_DISCOUNTS = {
    "name": "lookup_discounts",
    "description": "Look up discounts and offers relevant to the traveler's profile (student, youth, senior, etc.).",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The destination city"},
            "user_profile": {
                "type": "string",
                "description": "Description of the traveler (e.g. 'university student', '68-year-old retiree')",
            },
        },
        "required": ["city", "user_profile"],
    },
}

FINISH_PLANNING = {
    "name": "finish_planning",
    "description": "Signal that planning is complete and trigger export of all files.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

# Which tools are available in each phase
TOOLS_BY_PHASE: dict[TripPhase, list[dict]] = {
    TripPhase.GATHERING:    [],
    TripPhase.SEARCHING:    [SEARCH_ATTRACTIONS],
    TripPhase.RECOMMENDING: [SEARCH_ATTRACTIONS],
    TripPhase.REFINING:     [SEARCH_ATTRACTIONS],
    TripPhase.HOSTEL:       [FIND_HOSTELS],
    TripPhase.DISCOUNT:     [LOOKUP_DISCOUNTS],
    TripPhase.EXPORT:       [FINISH_PLANNING],
    TripPhase.DONE:         [],
}
