import os
from bonvoyage.models.trip_state import TripState, TripPhase
from bonvoyage.models.tool_schemas import TOOLS_BY_PHASE
from bonvoyage.logging.tracer import log_event
from bonvoyage.tools.attraction_search import search_attractions
from bonvoyage.tools.hostel_search import find_hostels
from bonvoyage.tools.discount_lookup import lookup_discounts
from bonvoyage.tools.exporter import export_csv, generate_markdown_summary


def dispatch(tool_name: str, tool_args: dict, state: TripState) -> tuple[str, bool]:
    """
    Execute a tool call and update TripState.
    Returns (observation_text, is_finish_planning).
    """
    state.advance_step()

    log_event(
        session_id=state.session_id,
        step=state.step,
        phase=state.phase.value,
        event="tool_call",
        tool=tool_name,
        input=tool_args,
    )

    if tool_name == "search_attractions":
        results = search_attractions(
            city=tool_args.get("city", state.destination),
            interests=tool_args.get("interests", state.interests),
            must_visit=tool_args.get("must_visit", state.must_visit),
        )
        state.selected_attractions = results
        if state.phase == TripPhase.SEARCHING:
            state.phase = TripPhase.RECOMMENDING

        observation = _format_attractions(results)
        log_event(
            session_id=state.session_id,
            step=state.step,
            phase=state.phase.value,
            event="tool_result",
            tool=tool_name,
            output={"count": len(results)},
        )
        return observation, False

    if tool_name == "find_hostels":
        results = find_hostels(city=tool_args.get("city", state.destination))
        state.selected_hostels = results
        state.phase = TripPhase.DISCOUNT

        observation = _format_hostels(results)
        log_event(
            session_id=state.session_id,
            step=state.step,
            phase=state.phase.value,
            event="tool_result",
            tool=tool_name,
            output={"count": len(results)},
        )
        return observation, False

    if tool_name == "lookup_discounts":
        results = lookup_discounts(
            city=tool_args.get("city", state.destination),
            user_profile=tool_args.get("user_profile", state.user_profile),
        )
        state.discounts = results
        state.phase = TripPhase.EXPORT

        observation = _format_discounts(results)
        log_event(
            session_id=state.session_id,
            step=state.step,
            phase=state.phase.value,
            event="tool_result",
            tool=tool_name,
            output={"count": len(results)},
        )
        return observation, False

    if tool_name == "finish_planning":
        attractions_path = export_csv(state.selected_attractions, "attractions", state.session_id)
        hostels_path = export_csv(state.selected_hostels, "hostels", state.session_id)
        summary_path = generate_markdown_summary(state)
        state.phase = TripPhase.DONE

        log_event(
            session_id=state.session_id,
            step=state.step,
            phase=state.phase.value,
            event="export_complete",
            files=[attractions_path, hostels_path, summary_path],
        )
        return (
            f"Exports complete:\n- {attractions_path}\n- {hostels_path}\n- {summary_path}",
            True,
        )

    return f"Unknown tool: {tool_name}", False


def _format_attractions(attractions) -> str:
    if not attractions:
        return "No attractions found."
    lines = []
    for i, a in enumerate(attractions, 1):
        lines.append(f"{i}. **{a.name}** — {a.description}\n   *Why:* {a.rationale}")
    return "\n\n".join(lines)


def _format_hostels(hostels) -> str:
    if not hostels:
        return "No hostels found."
    lines = []
    for i, h in enumerate(hostels, 1):
        lines.append(f"{i}. **{h.name}** — {h.description}")
    return "\n\n".join(lines)


def _format_discounts(discounts) -> str:
    if not discounts:
        return "No specific discounts found for your profile."
    lines = []
    for d in discounts:
        lines.append(f"- **{d.name}**: {d.description} *(Eligibility: {d.eligibility})*")
    return "\n".join(lines)
