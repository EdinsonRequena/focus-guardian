"""Audio placeholders for the first project scaffold."""

from collections.abc import Sequence
import logging

LOGGER = logging.getLogger(__name__)


class AudioPlayer:
    """Placeholder audio boundary for future pygame integration."""

    def __init__(self, sound_paths: Sequence[str]) -> None:
        self.sound_paths = tuple(sound_paths)

    def play_alert(self) -> None:
        """Log a placeholder action without loading or playing audio yet."""
        LOGGER.debug(
            "Audio placeholder invoked with %s configured sound(s).",
            len(self.sound_paths),
        )
