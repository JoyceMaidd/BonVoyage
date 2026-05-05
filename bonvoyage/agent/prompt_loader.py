from pathlib import Path
import yaml

from bonvoyage.models.trip_state import TripPhase


_prompts: dict | None = None
_prompt_version: str = ""


def _load() -> dict:
    global _prompts, _prompt_version
    if _prompts is None:
        path = Path(__file__).parent.parent / "prompts" / "system_v1.yaml"
        with open(path, encoding="utf-8") as f:
            _prompts = yaml.safe_load(f)
        _prompt_version = _prompts.get("version", "unknown")
    return _prompts


def get_prompt_version() -> str:
    _load()
    return _prompt_version


def build_system_prompt(phase: TripPhase) -> str:
    data = _load()
    base = data["system"].strip()
    addendum = data.get("phase_addenda", {}).get(phase.value, "")
    if addendum:
        return f"{base}\n\n## Current phase: {phase.value}\n{addendum.strip()}"
    return base
