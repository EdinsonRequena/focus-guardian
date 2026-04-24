"""Audio playback with a silent fallback for the MVP."""

from collections.abc import Sequence
import logging
from pathlib import Path

from core.settings import BASE_DIR
from features.state import DistractionReason

LOGGER = logging.getLogger(__name__)


class AudioPlayer:
    """Play alert sounds when available, otherwise stay silent safely."""

    def __init__(self, sound_paths: Sequence[str], volume: float = 0.8) -> None:
        self.sound_paths = tuple(sound_paths)
        self.volume = max(0.0, min(volume, 1.0))
        self.sound_index = 0
        self.silent_mode = False
        self._silent_mode_logged = False
        self.sounds: list[object] = []
        self.music_paths: list[Path] = []
        self._pygame = None
        self._initialize()

    def _initialize(self) -> None:
        configured_paths = [
            (sound_path, (BASE_DIR / sound_path).resolve())
            for sound_path in self.sound_paths
            if sound_path
        ]
        existing_paths = [resolved_path for _, resolved_path in configured_paths if resolved_path.exists()]
        missing_paths = [configured_path for configured_path, resolved_path in configured_paths if not resolved_path.exists()]

        LOGGER.info("Audio configured paths: %s", list(self.sound_paths) or ["none"])
        if missing_paths:
            LOGGER.info("Audio paths ignored because they do not exist: %s", missing_paths)
        LOGGER.info("Audio existing paths: %s", [str(path) for path in existing_paths] or ["none"])

        if not existing_paths:
            self.silent_mode = True
            self._log_silent_mode("No sound files were found.")
            return

        try:
            import pygame

            self._pygame = pygame
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            LOGGER.info("Audio mixer initialized successfully.")
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
                    sound.set_volume(self.volume)
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
            "Audio initialized successfully. Loaded %s audio file(s) at volume %.2f.",
            loaded_count,
            self.volume,
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
                self._pygame.mixer.music.set_volume(self.volume)
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
