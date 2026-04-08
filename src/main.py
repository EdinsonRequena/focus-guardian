"""Application entry point for Focus Guardian."""

import logging

from core.settings import load_settings
from features.audio import AudioPlayer
from features.detector import FocusDetector
from features.service import FocusGuardianService
from features.ui import FocusUI


def configure_logging() -> None:
    """Set a small default logging configuration for local runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Bootstrap the app with placeholder dependencies."""
    configure_logging()

    settings = load_settings()
    service = FocusGuardianService(
        settings=settings,
        detector=FocusDetector(),
        audio_player=AudioPlayer(sound_paths=settings.sound_paths),
        ui=FocusUI(window_title=settings.window_title),
    )
    service.run()


if __name__ == "__main__":
    main()
