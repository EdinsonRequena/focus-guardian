"""Small application coordinator for the project scaffold."""

import logging

from core.settings import Settings
from features.audio import AudioPlayer
from features.detector import FocusDetector
from features.state import FocusState, FocusStatus
from features.ui import FocusUI

LOGGER = logging.getLogger(__name__)


class FocusGuardianService:
    """Coordinate placeholders without implementing product behavior."""

    def __init__(
        self,
        settings: Settings,
        detector: FocusDetector,
        audio_player: AudioPlayer,
        ui: FocusUI,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.audio_player = audio_player
        self.ui = ui

    def run(self) -> None:
        """Start the scaffold and expose the future integration points."""
        status = FocusStatus(
            state=FocusState.UNKNOWN,
            reason="project_scaffold",
        )
        self.ui.render(status)
        LOGGER.info(
            "%s scaffold ready. Detection, state transitions, audio, and overlay "
            "behavior are intentionally pending.",
            self.settings.app_name,
        )
