"""Face detection and heuristic extraction for the live MVP."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_module

from core.settings import BASE_DIR, Settings
from features.state import DistractionReason

LOGGER = logging.getLogger(__name__)

NOSE_TIP_INDEX = 1
LEFT_CHEEK_INDEX = 234
RIGHT_CHEEK_INDEX = 454
LEFT_EYE_INDEX = 33
RIGHT_EYE_INDEX = 263
MOUTH_CENTER_INDEX = 13
CHIN_INDEX = 152

LANDMARK_CONNECTIONS = [
    *[
        (connection.start, connection.end)
        for connection in face_landmarker_module.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
    ],
    *[
        (connection.start, connection.end)
        for connection in face_landmarker_module.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
    ],
    *[
        (connection.start, connection.end)
        for connection in face_landmarker_module.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS
    ],
]


@dataclass(slots=True)
class DetectionSnapshot:
    """Detector output consumed by service and UI."""

    has_face: bool
    raw_reason: DistractionReason | None
    face_box: tuple[int, int, int, int] | None = None
    landmark_points: list[tuple[int, int]] = field(default_factory=list)
    landmark_connections: list[tuple[int, int]] = field(default_factory=list)
    debug_values: dict[str, str] = field(default_factory=dict)


class FocusDetector:
    """Extract face presence and simple attention heuristics from a frame."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = "face_landmarker_task"
        self._smoothed_turn_ratio = 0.0
        self._smoothed_down_ratio = 0.0
        self._last_timestamp_ms = 0

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.landmarker = self._create_face_landmarker()

    def analyze(
        self,
        frame: np.ndarray,
        now_seconds: float | None = None,
    ) -> DetectionSnapshot:
        """Analyze a frame and return a simple observation."""
        analysis_frame, scale_x, scale_y = self._prepare_analysis_frame(frame)
        timestamp_ms = self._to_timestamp_ms(now_seconds)

        if self.landmarker is not None:
            return self._analyze_with_tasks(
                analysis_frame=analysis_frame,
                scale_x=scale_x,
                scale_y=scale_y,
                timestamp_ms=timestamp_ms,
            )

        return self._analyze_with_opencv_fallback(
            analysis_frame=analysis_frame,
            scale_x=scale_x,
            scale_y=scale_y,
        )

    def close(self) -> None:
        """Release detector resources."""
        if self.landmarker is not None:
            self.landmarker.close()

    def _create_face_landmarker(self) -> object | None:
        model_path = (BASE_DIR / self.settings.face_landmarker_model_path).resolve()
        if not model_path.exists():
            LOGGER.warning(
                "Face Landmarker model not found at %s. Falling back to OpenCV.",
                model_path,
            )
            self.backend = "opencv_fallback"
            return None

        try:
            base_options = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=self.settings.min_detection_confidence,
                min_face_presence_confidence=self.settings.min_face_presence_confidence,
                min_tracking_confidence=self.settings.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            LOGGER.info("Using MediaPipe Face Landmarker Task backend.")
            return vision.FaceLandmarker.create_from_options(options)
        except Exception as error:  # pragma: no cover - runtime fallback
            LOGGER.warning(
                "Face Landmarker Task init failed (%s). Falling back to OpenCV.",
                error,
            )
            self.backend = "opencv_fallback"
            return None

    def _analyze_with_tasks(
        self,
        analysis_frame: np.ndarray,
        scale_x: float,
        scale_y: float,
        timestamp_ms: int,
    ) -> DetectionSnapshot:
        rgb_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            self._smooth_metrics(0.0, 0.0)
            return self._no_face_snapshot()

        face_landmarks = result.face_landmarks[0]
        analysis_height, analysis_width = analysis_frame.shape[:2]
        analysis_points = self._landmarks_to_points(
            face_landmarks,
            analysis_width,
            analysis_height,
        )
        display_points = self._scale_points(analysis_points, scale_x, scale_y)
        face_box = self._points_to_box(display_points)

        turn_ratio = self._compute_turn_ratio(face_landmarks)
        down_ratio = self._compute_down_ratio(face_landmarks)
        turn_ratio, down_ratio = self._smooth_metrics(turn_ratio, down_ratio)
        raw_reason = self._reason_from_metrics(turn_ratio, down_ratio)

        return DetectionSnapshot(
            has_face=True,
            raw_reason=raw_reason,
            face_box=face_box,
            landmark_points=display_points,
            landmark_connections=LANDMARK_CONNECTIONS,
            debug_values={
                "backend": self.backend,
                "face_detected": "yes",
                "raw_reason": raw_reason.value if raw_reason else "focused",
                "turn_ratio": f"{turn_ratio:.3f}",
                "down_ratio": f"{down_ratio:.3f}",
                "analysis": f"{analysis_width}x{analysis_height}",
            },
        )

    def _analyze_with_opencv_fallback(
        self,
        analysis_frame: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> DetectionSnapshot:
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(70, 70),
        )
        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(70, 70),
        )

        if len(faces) == 0 and len(profiles) == 0:
            self._smooth_metrics(0.0, 0.0)
            return self._no_face_snapshot()

        if len(faces) > 0:
            face_box = self._largest_box([tuple(map(int, face)) for face in faces])
            display_box = self._scale_box(face_box, scale_x, scale_y)
            self._smooth_metrics(0.0, 0.0)
            return DetectionSnapshot(
                has_face=True,
                raw_reason=None,
                face_box=display_box,
                debug_values={
                    "backend": self.backend,
                    "face_detected": "yes",
                    "raw_reason": "focused",
                    "turn_ratio": "0.000",
                    "down_ratio": "0.000",
                    "analysis": f"{gray.shape[1]}x{gray.shape[0]}",
                },
            )

        profile_box = self._largest_box([tuple(map(int, face)) for face in profiles])
        display_box = self._scale_box(profile_box, scale_x, scale_y)
        self._smooth_metrics(0.0, 0.0)
        return DetectionSnapshot(
            has_face=True,
            raw_reason=DistractionReason.LOOKING_AWAY,
            face_box=display_box,
            debug_values={
                "backend": self.backend,
                "face_detected": "yes",
                "raw_reason": DistractionReason.LOOKING_AWAY.value,
                "turn_ratio": "profile",
                "down_ratio": "0.000",
                "analysis": f"{gray.shape[1]}x{gray.shape[0]}",
            },
        )

    def _prepare_analysis_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= self.settings.analysis_frame_width:
            return frame, 1.0, 1.0

        scale = self.settings.analysis_frame_width / frame_width
        analysis_height = max(1, int(frame_height * scale))
        analysis_frame = cv2.resize(
            frame,
            (self.settings.analysis_frame_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )
        return (
            analysis_frame,
            frame_width / analysis_frame.shape[1],
            frame_height / analysis_frame.shape[0],
        )

    def _to_timestamp_ms(self, now_seconds: float | None) -> int:
        if now_seconds is None:
            now_seconds = time.monotonic()

        timestamp_ms = int(now_seconds * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _reason_from_metrics(
        self,
        turn_ratio: float,
        down_ratio: float,
    ) -> DistractionReason | None:
        if turn_ratio >= self.settings.looking_away_ratio_threshold:
            return DistractionReason.LOOKING_AWAY
        if down_ratio >= self.settings.looking_down_ratio_threshold:
            return DistractionReason.LOOKING_DOWN
        return None

    def _smooth_metrics(self, turn_ratio: float, down_ratio: float) -> tuple[float, float]:
        alpha = self.settings.metric_smoothing
        self._smoothed_turn_ratio = ((1 - alpha) * self._smoothed_turn_ratio) + (
            alpha * turn_ratio
        )
        self._smoothed_down_ratio = ((1 - alpha) * self._smoothed_down_ratio) + (
            alpha * down_ratio
        )
        return self._smoothed_turn_ratio, self._smoothed_down_ratio

    def _no_face_snapshot(self) -> DetectionSnapshot:
        return DetectionSnapshot(
            has_face=False,
            raw_reason=DistractionReason.NO_FACE,
            debug_values={
                "backend": self.backend,
                "face_detected": "no",
                "raw_reason": DistractionReason.NO_FACE.value,
                "turn_ratio": "0.000",
                "down_ratio": "0.000",
                "analysis": "n/a",
            },
        )

    @staticmethod
    def _landmarks_to_points(
        landmarks: list[object],
        frame_width: int,
        frame_height: int,
    ) -> list[tuple[int, int]]:
        return [
            (
                min(max(int(landmark.x * frame_width), 0), frame_width - 1),
                min(max(int(landmark.y * frame_height), 0), frame_height - 1),
            )
            for landmark in landmarks
        ]

    @staticmethod
    def _points_to_box(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _scale_points(
        points: list[tuple[int, int]],
        scale_x: float,
        scale_y: float,
    ) -> list[tuple[int, int]]:
        return [
            (int(point[0] * scale_x), int(point[1] * scale_y))
            for point in points
        ]

    @staticmethod
    def _scale_box(
        face_box: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int, int, int]:
        x, y, width, height = face_box
        return (
            int(x * scale_x),
            int(y * scale_y),
            int((x + width) * scale_x),
            int((y + height) * scale_y),
        )

    @staticmethod
    def _largest_box(
        boxes: list[tuple[int, int, int, int]],
    ) -> tuple[int, int, int, int]:
        return max(boxes, key=lambda item: item[2] * item[3])

    @staticmethod
    def _compute_turn_ratio(landmarks: list[object]) -> float:
        nose = landmarks[NOSE_TIP_INDEX]
        left_cheek = landmarks[LEFT_CHEEK_INDEX]
        right_cheek = landmarks[RIGHT_CHEEK_INDEX]
        left_eye = landmarks[LEFT_EYE_INDEX]
        right_eye = landmarks[RIGHT_EYE_INDEX]

        face_center_x = (left_cheek.x + right_cheek.x) / 2
        face_width = max(right_cheek.x - left_cheek.x, 1e-6)
        nose_offset = abs(nose.x - face_center_x) / face_width

        left_span = max(nose.x - left_cheek.x, 1e-6)
        right_span = max(right_cheek.x - nose.x, 1e-6)
        cheek_asymmetry = abs(left_span - right_span) / face_width

        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_offset = abs(nose.x - eye_center_x) / face_width

        return max(nose_offset * 1.2, cheek_asymmetry, eye_offset)

    @staticmethod
    def _compute_down_ratio(landmarks: list[object]) -> float:
        left_eye = landmarks[LEFT_EYE_INDEX]
        right_eye = landmarks[RIGHT_EYE_INDEX]
        nose = landmarks[NOSE_TIP_INDEX]
        mouth = landmarks[MOUTH_CENTER_INDEX]
        chin = landmarks[CHIN_INDEX]

        eye_center_y = (left_eye.y + right_eye.y) / 2
        lower_face_span = max(chin.y - eye_center_y, 1e-6)
        nose_ratio = (nose.y - eye_center_y) / lower_face_span
        mouth_ratio = (mouth.y - eye_center_y) / lower_face_span
        return max(0.0, ((nose_ratio * 0.6) + (mouth_ratio * 0.4)) - 0.22)
