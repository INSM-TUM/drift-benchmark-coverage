"""Event and Trace models for XES event log parsing."""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Represents a single event in a process trace.

    Attributes:
        activity: Activity name (concept:name in XES)
        timestamp: Optional timestamp (time:timestamp in XES)
    """

    activity: str = Field(description="Activity name")
    timestamp: Optional[datetime] = Field(default=None, description="Event timestamp")

    model_config = {
        "frozen": True,  # Immutable events
    }


class Trace(BaseModel):
    """Represents a process trace (case) containing ordered events.

    Attributes:
        case_id: Case identifier (concept:name in XES trace)
        events: Ordered list of events in this trace
    """

    case_id: str = Field(description="Case identifier")
    events: List[Event] = Field(default_factory=list, description="Ordered events")

    model_config = {
        "frozen": True,  # Immutable traces
    }
