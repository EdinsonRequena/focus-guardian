"""Minimal state types for the first project scaffold."""

from dataclasses import dataclass
from enum import Enum


class FocusState(str, Enum):
    """High-level states reserved for future behavior."""

    UNKNOWN = "unknown"
    FOCUSED = "focused"
    DISTRACTED = "distracted"


@dataclass(slots=True)
class FocusStatus:
    """Current status presented to other local modules."""

    state: FocusState = FocusState.UNKNOWN
    reason: str = "initial"
