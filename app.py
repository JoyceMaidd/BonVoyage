import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt

from bonvoyage.models.trip_state import TripState, TripPhase
from bonvoyage.models.tool_schemas import TOOLS_BY_PHASE
from bonvoyage.agent.prompt_loader import build_system_prompt, get_prompt_version
from bonvoyage.agent.intent_extractor import extract_trip_intent, build_missing_fields_prompt
from bonvoyage.agent.controller import dispatch
from bonvoyage.logging_custom.tracer import log_event
from bonvoyage.tools.exporter import export_csv, generate_markdown_summary


# ── Gemini setup ──────────────────────────────────────────────────────────────

def _configure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        genai.configure(api_key=api_key)
    return api_key


def _make_tool_config(tools: list[dict]) -> list:
    if not tools:
        return []
    return [genai.protos.Tool(function_declarations=[
        genai.protos.FunctionDeclaration(**t) for t in tools
    ])]


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
def _call_gemini(model, history, system_prompt, tools):
    delay = float(os.environ.get("GEMINI_CALL_DELAY", "0"))
    if delay > 0:
        import time
        time.sleep(delay)
    return model.generate_content(
        history,
        system_instruction=system_prompt,
        tools=_make_tool_config(tools) if tools else None,
    )


# ── Session state init ────────────────────────────────────────────────────────

def _init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "trip_state" not in st.session_state:
        st.session_state.trip_state = TripState(session_id=st.session_state.session_id)
    if "history" not in st.session_state:
        st.session_state.history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "awaiting_input" not in st.session_state:
        st.session_state.awaiting_input = True
    if "exports" not in st.session_state:
        st.session_state.exports = {}
    if "loop_running" not in st.session_state:
        st.session_state.loop_running = False
    if "trace_log" not in st.session_state:
        st.session_state.trace_log = []


def _add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


def _log(session_id, step, phase, event, **kwargs):
    log_event(session_id, step, phase, event, **kwargs)
    st.session_state.trace_log.append({"step": step, "phase": phase, "event": event, **kwargs})


# ── Agent step ────────────────────────────────────────────────────────────────

def _run_agent_turn(user_input: str):
    """Process one user message through the agent. May do multiple tool calls before waiting."""
    state: TripState = st.session_state.trip_state
    history: list = st.session_state.history
    session_id = st.session_state.session_id

    _add_message("user", user_input)
    history.append({"role": "user", "parts": [user_input]})
    _log(session_id, state.step, state.phase.value, "user_input", text=user_input)

    # --- Intent extraction on first message ---
    if state.phase == TripPhase.GATHERING:
        with st.spinner("Understanding your trip request..."):
            intent = extract_trip_intent(user_input)

        state.destination = intent.destination or state.destination
        state.duration_days = intent.duration_days or state.duration_days
        state.user_profile = intent.user_profile or state.user_profile
        state.interests = intent.interests or state.interests
        state.must_visit = intent.must_visit or state.must_visit

        _log(session_id, 0, "GATHERING", "intent_extracted",
             destination=state.destination, duration_days=state.duration_days)

        if intent.missing_fields:
            follow_up = build_missing_fields_prompt(intent)
            _add_message("assistant", follow_up)
            history.append({"role": "model", "parts": [follow_up]})
            return  # Wait for user to provide missing info

        if not state.is_intent_complete():
            _add_message("assistant", "I couldn't determine your destination or trip length. Could you rephrase?")
            return

        state.phase = TripPhase.SEARCHING
        confirmation = (
            f"Got it! Planning a **{state.duration_days}-day trip to {state.destination}** "
            f"for a _{state.user_profile or 'solo traveler'}_ interested in "
            f"_{', '.join(state.interests) or 'general sightseeing'}_."
            + (f" Must visit: _{', '.join(state.must_visit)}_." if state.must_visit else "")
            + "\n\nLet me find the best attractions for you..."
        )
        _add_message("assistant", confirmation)
        history.append({"role": "model", "parts": [confirmation]})

    # --- Advance phase based on user signals (for text-response turns) ---
    state.phase = _maybe_advance_phase(state, user_input)

    # --- ReAct inner loop: keep calling Gemini until it produces a text response ---
    model = genai.GenerativeModel("gemini-1.5-flash")

    for _ in range(6):  # Safety cap on back-to-back tool calls
        if state.phase == TripPhase.DONE:
            break

        tools = TOOLS_BY_PHASE[state.phase]
        system_prompt = build_system_prompt(state.phase)

        _log(session_id, state.step, state.phase.value, "llm_call",
             tools=[t["name"] for t in tools])

        with st.spinner("Thinking..."):
            response = _call_gemini(model, history, system_prompt, tools)

        candidate = response.candidates[0]
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

            _log(session_id, state.step, state.phase.value, "function_call",
                 tool=tool_name, args=str(tool_args))

            with st.spinner(f"Running {tool_name.replace('_', ' ')}..."):
                observation, is_done = dispatch(tool_name, tool_args, state)

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
                # Collect export paths
                sid = state.session_id
                exports_dir = Path(__file__).parent / "exports"
                st.session_state.exports = {
                    "attractions": exports_dir / f"{sid}_attractions.csv",
                    "hostels": exports_dir / f"{sid}_hostels.csv",
                    "summary": exports_dir / f"{sid}_trip_summary.md",
                }
                _add_message("assistant",
                    f"Your trip plan for **{state.destination}** is ready! "
                    "Download your files below.")
                break

        elif text_parts:
            agent_text = " ".join(text_parts)
            _add_message("assistant", agent_text)
            history.append({"role": "model", "parts": [agent_text]})
            break

        else:
            break


def _maybe_advance_phase(state: TripState, user_input: str) -> TripPhase:
    lower = user_input.lower()
    affirmative = any(w in lower for w in [
        "yes", "ok", "okay", "proceed", "continue", "looks good",
        "perfect", "great", "done", "ready", "sure", "go ahead", "next",
        "all good", "happy", "confirmed", "let's go", "sounds good",
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(state: TripState):
    with st.sidebar:
        st.title("BonVoyage")
        st.caption(f"Session: `{state.session_id}` · Prompt: `v{get_prompt_version()}`")

        st.divider()
        st.subheader("Trip Details")

        if state.destination:
            st.markdown(f"**Destination:** {state.destination}")
        if state.duration_days:
            st.markdown(f"**Duration:** {state.duration_days} days")
        if state.user_profile:
            st.markdown(f"**Profile:** {state.user_profile}")
        if state.interests:
            st.markdown(f"**Interests:** {', '.join(state.interests)}")
        if state.must_visit:
            st.markdown(f"**Must visit:** {', '.join(state.must_visit)}")

        phase_colors = {
            TripPhase.GATHERING: "🔵",
            TripPhase.SEARCHING: "🟡",
            TripPhase.RECOMMENDING: "🟠",
            TripPhase.REFINING: "🟠",
            TripPhase.HOSTEL: "🟣",
            TripPhase.DISCOUNT: "🟢",
            TripPhase.EXPORT: "🔴",
            TripPhase.DONE: "✅",
        }
        st.markdown(f"**Phase:** {phase_colors.get(state.phase, '⚪')} {state.phase.value}")

        if state.selected_attractions:
            st.divider()
            st.subheader(f"Attractions ({len(state.selected_attractions)})")
            for i, a in enumerate(state.selected_attractions, 1):
                st.markdown(f"{i}. {a.name}")

        if state.selected_hostels:
            st.divider()
            st.subheader(f"Hostels ({len(state.selected_hostels)})")
            for h in state.selected_hostels:
                st.markdown(f"• {h.name}")

        # Exports
        if st.session_state.exports:
            st.divider()
            st.subheader("Downloads")
            for label, path in [
                ("attractions.csv", "attractions"),
                ("hostels.csv", "hostels"),
                ("trip_summary.md", "summary"),
            ]:
                file_path = st.session_state.exports.get(path)
                if file_path and Path(file_path).exists():
                    st.download_button(
                        label=f"⬇ {label}",
                        data=Path(file_path).read_bytes(),
                        file_name=label,
                        key=f"dl_{path}",
                    )

        # Debug trace
        if st.session_state.trace_log:
            with st.expander("Agent trace", expanded=False):
                for entry in st.session_state.trace_log[-15:]:
                    st.json(entry)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="BonVoyage", page_icon="✈️", layout="wide")

    api_key = _configure_gemini()
    if not api_key:
        st.error("GEMINI_API_KEY not set. Add it to your .env file.")
        st.stop()

    _init_session()
    state: TripState = st.session_state.trip_state

    _render_sidebar(state)

    st.title("✈️ BonVoyage")
    st.caption("Your AI travel planning assistant for solo travelers")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Initial prompt
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            welcome = (
                "Hi! I'm BonVoyage. Tell me about your trip and I'll plan it for you.\n\n"
                "For example: *\"I'm a university student doing a 4-day solo trip in Paris. "
                "I love arts and ballet. I must visit Opéra de Paris.\"*"
            )
            st.markdown(welcome)
            st.session_state.messages.append({"role": "assistant", "content": welcome})

    # Chat input
    if user_input := st.chat_input("Describe your trip..."):
        _run_agent_turn(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
