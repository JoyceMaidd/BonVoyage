from enum import Enum
from pydantic import BaseModel, Field


class TripPhase(str, Enum):
    GATHERING    = "GATHERING"
    SEARCHING    = "SEARCHING"
    RECOMMENDING = "RECOMMENDING"
    REFINING     = "REFINING"
    HOSTEL       = "HOSTEL"
    DISCOUNT     = "DISCOUNT"
    EXPORT       = "EXPORT"
    DONE         = "DONE"


class Attraction(BaseModel):
    name: str
    description: str
    address: str = ""
    rationale: str = ""
    lat: float | None = None
    lon: float | None = None


class Hostel(BaseModel):
    name: str
    description: str
    address: str = ""


class Discount(BaseModel):
    name: str
    description: str
    eligibility: str = ""


class TripState(BaseModel):
    session_id: str
    destination: str = ""
    duration_days: int = 0
    user_profile: str = ""
    interests: list[str] = Field(default_factory=list)
    must_visit: list[str] = Field(default_factory=list)
    selected_attractions: list[Attraction] = Field(default_factory=list)
    selected_hostels: list[Hostel] = Field(default_factory=list)
    discounts: list[Discount] = Field(default_factory=list)
    phase: TripPhase = TripPhase.GATHERING
    step: int = 0

    def is_intent_complete(self) -> bool:
        return bool(self.destination and self.duration_days > 0)

    def advance_step(self) -> None:
        self.step += 1
