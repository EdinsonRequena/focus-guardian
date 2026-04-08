"""Project settings and environment loading."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


@dataclass(slots=True)
class Settings:
    """Runtime settings for the local MVP scaffold."""

    app_name: str
    camera_index: int
    window_title: str
    frame_width: int
    frame_height: int
    no_face_threshold_seconds: float
    looking_away_threshold_seconds: float
    looking_down_threshold_seconds: float
    recovery_threshold_seconds: float
    audio_cooldown_seconds: float
    show_debug_metrics: bool
    show_landmarks: bool
    sound_paths: tuple[str, ...]


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Load settings from the local environment without adding app logic."""
    load_dotenv(ENV_FILE)

    sound_paths = tuple(
        path
        for path in (
            os.getenv("SOUND_1_PATH"),
            os.getenv("SOUND_2_PATH"),
            os.getenv("SOUND_3_PATH"),
        )
        if path
    )

    return Settings(
        app_name="Focus Guardian",
        camera_index=int(os.getenv("CAMERA_INDEX", "0")),
        window_title=os.getenv("WINDOW_TITLE", "Focus Guardian"),
        frame_width=int(os.getenv("FRAME_WIDTH", "1280")),
        frame_height=int(os.getenv("FRAME_HEIGHT", "720")),
        no_face_threshold_seconds=float(
            os.getenv("NO_FACE_THRESHOLD_SECONDS", "0.8")
        ),
        looking_away_threshold_seconds=float(
            os.getenv("LOOKING_AWAY_THRESHOLD_SECONDS", "1.0")
        ),
        looking_down_threshold_seconds=float(
            os.getenv("LOOKING_DOWN_THRESHOLD_SECONDS", "1.0")
        ),
        recovery_threshold_seconds=float(
            os.getenv("RECOVERY_THRESHOLD_SECONDS", "0.6")
        ),
        audio_cooldown_seconds=float(os.getenv("AUDIO_COOLDOWN_SECONDS", "3.0")),
        show_debug_metrics=_get_bool("SHOW_DEBUG_METRICS", False),
        show_landmarks=_get_bool("SHOW_LANDMARKS", False),
        sound_paths=sound_paths,
    )
