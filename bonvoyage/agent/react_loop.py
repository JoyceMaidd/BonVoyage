import os
import time
import uuid
from collections.abc import Generator
from google import genai
from tenacity import retry, wait_exponential, stop_after_attempt

from bonvoyage.models.trip_state import TripState, TripPhase
from bonvoyage.models.tool_schemas import TOOLS_BY_PHASE
from bonvoyage.agent.prompt_loader import build_system_prompt, get_prompt_version
from bonvoyage.agent.intent_extractor import extract_trip_intent, build_missing_fields_prompt
from bonvoyage.agent.controller import dispatch
from bonvoyage.logging_custom.tracer import log_event
from bonvoyage.gemini_client import get_client


def _make_tool_config(tools: list[dict]) -> list:
    if not tools:
        return []
    return [genai.protos.Tool(function_declarations=[
        genai.protos.FunctionDeclaration(**t) for t in tools
    ])]


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3), reraise=True)
def _call_gemini(client: genai.Client, history: list, system_prompt: str, tools: list):
    """Call Gemini API with system prompt and optional tools."""
    delay = float(os.environ.get("GEMINI_CALL_DELAY", "0"))
    if delay > 0:
        time.sleep(delay)
    return client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=history,
        system_instruction=system_prompt,
        tools=_make_tool_config(tools) if tools else None,
    )


def run_session(
    initial_input: str,
    on_message: callable = print,
) -> TripState:
    """
    Run a full BonVoyage planning session.

    Args:
        initial_input: The user's natural language trip request.
        on_message: Callback for agent text responses (default: print to stdout).

    Returns:
        Final TripState after session completes.
    """
    client = get_client()

    session_id = str(uuid.uuid4())[:8]

    log_event(session_id, 0, "INIT", "session_start",
              prompt_version=get_prompt_version(), initial_input=initial_input)

    # --- Intent extraction ---
    intent = extract_trip_intent(initial_input)
    state = TripState(
        session_id=session_id,
        destination=intent.destination,
        duration_days=intent.duration_days,
        user_profile=intent.user_profile,
        interests=intent.interests,
        must_visit=intent.must_visit,
    )

    log_event(session_id, 0, "GATHERING", "intent_extracted",
              destination=state.destination, duration_days=state.duration_days,
              interests=state.interests, must_visit=state.must_visit)

    # Ask one consolidated follow-up if required fields are missing
    if intent.missing_fields:
        follow_up = build_missing_fields_prompt(intent)
        on_message(follow_up)
        user_answer = input("> ").strip()

        # Re-extract with the combined context
        combined = f"{initial_input} {user_answer}"
        intent2 = extract_trip_intent(combined)
        state.destination = intent2.destination or state.destination
        state.duration_days = intent2.duration_days or state.duration_days
        state.user_profile = intent2.user_profile or state.user_profile
        state.interests = intent2.interests or state.interests
        state.must_visit = intent2.must_visit or state.must_visit

    if not state.is_intent_complete():
        on_message("I couldn't determine your destination or trip length. Please try again with more detail.")
        return state

    state.phase = TripPhase.SEARCHING

    # Build conversation history
    history = [
        {"role": "user", "parts": [initial_input]},
        {"role": "model", "parts": [
            f"Got it! Planning a {state.duration_days}-day trip to {state.destination} "
            f"for a {state.user_profile or 'solo traveler'} "
            f"interested in {', '.join(state.interests) or 'general sightseeing'}. "
            f"{'Must visit: ' + ', '.join(state.must_visit) + '. ' if state.must_visit else ''}"
            f"Let me find the best attractions for you."
        ]},
    ]

    # --- Main ReAct loop ---
    while state.phase != TripPhase.DONE:
        tools = TOOLS_BY_PHASE[state.phase]
        system_prompt = build_system_prompt(state.phase)

        log_event(session_id, state.step, state.phase.value, "llm_call",
                  tools=[t["name"] for t in tools])

        response = _call_gemini(client, history, system_prompt, tools)
        candidate = response.candidates[0]

        # Check for function call
        function_call = None
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                function_call = part.function_call
            elif hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if function_call:
            tool_name = function_call.name
            tool_args = dict(function_call.args)

            log_event(session_id, state.step, state.phase.value, "function_call",
                      tool=tool_name, args=tool_args)

            observation, is_done = dispatch(tool_name, tool_args, state)

            # Append tool call + result to history
            history.append({"role": "model", "parts": [
                genai.protos.Part(function_call=function_call)
            ]})
            history.append({"role": "user", "parts": [
                genai.protos.Part(function_response=genai.protos.FunctionResponse(
                    name=tool_name,
                    response={"result": observation},
                ))
            ]})

            if is_done:
                on_message(
                    f"\n✓ Your trip plan for {state.destination} is ready!\n\n"
                    f"Files saved to the exports/ folder. Check trip_summary.md for the full plan "
                    f"and import the CSV files to Google My Maps."
                )
                break

        elif text_parts:
            agent_text = " ".join(text_parts)
            on_message(agent_text)

            log_event(session_id, state.step, state.phase.value, "agent_message",
                      text=agent_text)

            history.append({"role": "model", "parts": [agent_text]})

            # Get user input and check for phase advancement triggers
            user_input = input("> ").strip()
            if not user_input:
                continue

            history.append({"role": "user", "parts": [user_input]})
            log_event(session_id, state.step, state.phase.value, "user_input",
                      text=user_input)

            # Advance phase based on user confirmation signals
            state.phase = _maybe_advance_phase(state, user_input)

        else:
            # Empty response — safety fallback
            log_event(session_id, state.step, state.phase.value, "empty_response")
            on_message("Something went wrong. Please try again.")
            break

    log_event(session_id, state.step, "DONE", "session_end",
              attractions=len(state.selected_attractions),
              hostels=len(state.selected_hostels))

    return state


def _maybe_advance_phase(state: TripState, user_input: str) -> TripPhase:
    """Advance phase if user signals they are done with the current step."""
    lower = user_input.lower()
    affirmative = any(w in lower for w in [
        "yes", "ok", "okay", "proceed", "continue", "looks good",
        "perfect", "great", "done", "ready", "sure", "go ahead", "next"
    ])

    if state.phase == TripPhase.RECOMMENDING and affirmative:
        return TripPhase.REFINING
    if state.phase == TripPhase.REFINING and affirmative:
        return TripPhase.HOSTEL
    if state.phase == TripPhase.HOSTEL and affirmative:
        return TripPhase.DISCOUNT
    if state.phase == TripPhase.DISCOUNT and affirmative:
        return TripPhase.EXPORT

    return state.phase
