"""Detector placeholders for the initial scaffold."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DetectionSnapshot:
    """Normalized detector output shape for future iterations."""

    has_face: bool = False
    is_focused: bool = False
    attention_score: float = 0.0


class FocusDetector:
    """Placeholder detector contract."""

    def analyze(self, frame: Any) -> DetectionSnapshot:
        """Return a neutral detection result until real vision logic exists."""
        _ = frame
        return DetectionSnapshot()
