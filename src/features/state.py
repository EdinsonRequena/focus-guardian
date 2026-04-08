"""State types for the live MVP session."""

from dataclasses import dataclass, field
from enum import Enum


class FocusState(str, Enum):
    """High-level application states."""

    FOCUSED = "focused"
    DISTRACTED = "distracted"


class DistractionReason(str, Enum):
    """Supported distraction reasons for the MVP."""

    NO_FACE = "no_face"
    LOOKING_AWAY = "looking_away"
    LOOKING_DOWN = "looking_down"


@dataclass(slots=True)
class SessionState:
    """Mutable state stored for the current run."""

    current_state: FocusState = FocusState.FOCUSED
    current_reason: DistractionReason | None = None
    distraction_count: int = 0
    candidate_reason: DistractionReason | None = None
    candidate_started_at: float | None = None
    stable_since: float | None = None
    last_audio_at: float | None = None
    active_episode_reason: DistractionReason | None = None


@dataclass(slots=True)
class SessionSnapshot:
    """UI-ready snapshot returned on every frame."""

    current_state: FocusState
    current_reason: DistractionReason | None
    distraction_count: int
    should_play_audio: bool = False
    audio_reason: DistractionReason | None = None
    candidate_reason: DistractionReason | None = None
    debug_values: dict[str, str] = field(default_factory=dict)
