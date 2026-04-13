"""Audio playback with a silent fallback for the MVP."""

from collections.abc import Sequence
import logging
from pathlib import Path

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
        self.music_paths: list[Path] = []
        self._pygame = None
        self._initialize()

    def _initialize(self) -> None:
        existing_paths = [
            (BASE_DIR / sound_path).resolve()
            for sound_path in self.sound_paths
            if sound_path and (BASE_DIR / sound_path).exists()
        ]
        if not existing_paths:
            self.silent_mode = True
            self._log_silent_mode("No sound files were found.")
            return

        try:
            import pygame

            self._pygame = pygame
            pygame.mixer.init()
        except Exception as error:  # pragma: no cover
            self.silent_mode = True
            self._log_silent_mode(f"Audio mixer init failed: {error}")
            return

        loaded_count = 0

        for path in existing_paths:
            suffix = path.suffix.lower()

            try:
                if suffix == ".mp3":
                    self.music_paths.append(path)
                    loaded_count += 1
                    LOGGER.info(
                        "Audio file registered as music stream: %s", path.name)
                else:
                    sound = self._pygame.mixer.Sound(str(path))
                    self.sounds.append(sound)
                    loaded_count += 1
                    LOGGER.info(
                        "Audio file loaded as sound effect: %s", path.name)
            except Exception as error:  # pragma: no cover
                LOGGER.warning(
                    "Failed to load audio file %s: %s", path.name, error)

        if loaded_count == 0:
            self.silent_mode = True
            self._log_silent_mode("No valid audio files could be loaded.")
            return

        LOGGER.info(
            "Audio initialized successfully. Loaded %s audio file(s).",
            loaded_count,
        )

    def play_distraction_alert(self, reason: DistractionReason | None) -> None:
        """Play the next alert if audio is available, otherwise do nothing."""
        _ = reason

        if self.silent_mode:
            self._log_silent_mode("Running without audio.")
            return

        try:
            if self.sounds:
                sound = self.sounds[self.sound_index % len(self.sounds)]
                self.sound_index += 1
                sound.play()
                return

            if self.music_paths:
                music_path = self.music_paths[self.sound_index % len(
                    self.music_paths)]
                self.sound_index += 1
                self._pygame.mixer.music.load(str(music_path))
                self._pygame.mixer.music.play()
                return

            self._log_silent_mode("No playable audio files are available.")
        except Exception as error:  # pragma: no cover
            self.silent_mode = True
            self.sounds = []
            self.music_paths = []
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
