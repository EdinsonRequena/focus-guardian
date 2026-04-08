"""Application entry point for Focus Guardian."""

import logging
import time

import cv2

from core.settings import load_settings
from features.audio import AudioPlayer
from features.detector import FocusDetector
from features.service import FocusGuardianService
from features.ui import FocusUI

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Set a small default logging configuration for local runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )


def create_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open the local webcam with the configured resolution."""
    camera = cv2.VideoCapture(camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def main() -> None:
    """Run the live local MVP."""
    configure_logging()

    settings = load_settings()
    detector = FocusDetector(settings)
    service = FocusGuardianService(settings)
    audio_player = AudioPlayer(sound_paths=settings.sound_paths)
    ui = FocusUI(settings)
    camera = create_camera(
        camera_index=settings.camera_index,
        width=settings.frame_width,
        height=settings.frame_height,
    )

    if not camera.isOpened():
        detector.close()
        audio_player.close()
        raise RuntimeError(
            f"Could not open camera index {settings.camera_index}. "
            "Check CAMERA_INDEX in your .env file."
        )

    LOGGER.info("Starting Focus Guardian. Press q to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                LOGGER.warning("Camera frame could not be read. Stopping session.")
                break

            if settings.mirror_preview:
                frame = cv2.flip(frame, 1)

            detection = detector.analyze(frame)
            session = service.update(detection, time.monotonic())
            if session.should_play_audio:
                audio_player.play_distraction_alert(session.audio_reason)

            rendered_frame = ui.render(frame, session, detection)
            cv2.imshow(settings.window_title, rendered_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        detector.close()
        audio_player.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
