"""Audio playback with a silent fallback for the MVP."""

from collections.abc import Sequence
import logging

from core.settings import BASE_DIR
from features.state import DistractionReason

LOGGER = logging.getLogger(__name__)


class AudioPlayer:
    """Play alert sounds when available, otherwise stay silent safely."""

    def __init__(self, sound_paths: Sequence[str]) -> None:
        self.sound_paths = tuple(sound_paths)
        self.sound_index = 0
        self.silent_mode = False
        self._silent_mode_logged = False
        self.sounds: list[object] = []
        self._pygame = None
        self._initialize()

    def _initialize(self) -> None:
        existing_paths = [
            (BASE_DIR / sound_path).resolve()
            for sound_path in self.sound_paths
            if (BASE_DIR / sound_path).exists()
        ]
        if not existing_paths:
            self.silent_mode = True
            self._log_silent_mode("No sound files were found.")
            return

        try:
            import pygame

            self._pygame = pygame
            pygame.mixer.init()
            self.sounds = [pygame.mixer.Sound(str(path)) for path in existing_paths]
        except Exception as error:  # pragma: no cover - defensive runtime path
            self.silent_mode = True
            self.sounds = []
            self._log_silent_mode(f"Audio init failed: {error}")

    def play_distraction_alert(self, reason: DistractionReason | None) -> None:
        """Play the next alert if audio is available, otherwise do nothing."""
        _ = reason

        if self.silent_mode or not self.sounds:
            self._log_silent_mode("Running without audio files.")
            return

        sound = self.sounds[self.sound_index % len(self.sounds)]
        self.sound_index += 1

        try:
            sound.play()
        except Exception as error:  # pragma: no cover - defensive runtime path
            self.silent_mode = True
            self.sounds = []
            self._log_silent_mode(f"Audio playback failed: {error}")

    def close(self) -> None:
        """Release mixer resources when they were initialized."""
        if self._pygame is not None and self._pygame.mixer.get_init():
            self._pygame.mixer.quit()

    def _log_silent_mode(self, reason: str) -> None:
        if self._silent_mode_logged:
            return

        LOGGER.info("Audio fallback enabled. %s", reason)
        self._silent_mode_logged = True
