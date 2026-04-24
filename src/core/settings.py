"""Project settings and environment loading."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


@dataclass(slots=True)
class Settings:
    """Runtime settings for the local MVP."""

    app_name: str
    camera_index: int
    window_title: str
    frame_width: int
    frame_height: int
    analysis_frame_width: int
    mirror_preview: bool
    face_landmarker_model_path: str
    no_face_threshold_seconds: float
    looking_away_threshold_seconds: float
    looking_down_threshold_seconds: float
    recovery_threshold_seconds: float
    audio_cooldown_seconds: float
    audio_volume: float
    looking_away_ratio_threshold: float
    looking_down_ratio_threshold: float
    eyes_down_ratio_threshold: float
    metric_smoothing: float
    min_detection_confidence: float
    min_face_presence_confidence: float
    min_tracking_confidence: float
    show_debug_metrics: bool
    show_landmarks: bool
    sound_paths: tuple[str, ...]


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_clamped_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    return max(minimum, min(value, maximum))


def load_settings() -> Settings:
    """Load local configuration from environment variables."""
    load_dotenv(ENV_FILE, override=True)

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
        analysis_frame_width=int(os.getenv("ANALYSIS_FRAME_WIDTH", "800")),
        mirror_preview=_get_bool("MIRROR_PREVIEW", True),
        face_landmarker_model_path=os.getenv(
            "FACE_LANDMARKER_MODEL_PATH", "assets/models/face_landmarker.task"
        ),
        no_face_threshold_seconds=float(
            os.getenv("NO_FACE_THRESHOLD_SECONDS", "0.8")
        ),
        looking_away_threshold_seconds=float(
            os.getenv("LOOKING_AWAY_THRESHOLD_SECONDS", "1.2")
        ),
        looking_down_threshold_seconds=float(
            os.getenv("LOOKING_DOWN_THRESHOLD_SECONDS", "0.35")
        ),
        recovery_threshold_seconds=float(
            os.getenv("RECOVERY_THRESHOLD_SECONDS", "0.35")
        ),
        audio_cooldown_seconds=float(
            os.getenv("AUDIO_COOLDOWN_SECONDS", "3.0")),
        audio_volume=_get_clamped_float("AUDIO_VOLUME", 0.8, 0.0, 1.0),
        looking_away_ratio_threshold=float(
            os.getenv("LOOKING_AWAY_RATIO_THRESHOLD", "0.24")
        ),
        looking_down_ratio_threshold=float(
            os.getenv("LOOKING_DOWN_RATIO_THRESHOLD", "0.26")
        ),
        eyes_down_ratio_threshold=float(
            os.getenv("EYES_DOWN_RATIO_THRESHOLD", "0.06")
        ),
        metric_smoothing=float(os.getenv("METRIC_SMOOTHING", "0.45")),
        min_detection_confidence=float(
            os.getenv("MIN_DETECTION_CONFIDENCE", "0.5")
        ),
        min_face_presence_confidence=float(
            os.getenv("MIN_FACE_PRESENCE_CONFIDENCE", "0.5")
        ),
        min_tracking_confidence=float(
            os.getenv("MIN_TRACKING_CONFIDENCE", "0.5")
        ),
        show_debug_metrics=_get_bool("SHOW_DEBUG_METRICS", False),
        show_landmarks=_get_bool("SHOW_LANDMARKS", True),
        sound_paths=sound_paths,
    )
