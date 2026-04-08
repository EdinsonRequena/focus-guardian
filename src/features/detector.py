"""Face detection and heuristic extraction for the live MVP."""

from dataclasses import dataclass, field
import logging

import cv2
import numpy as np

from core.settings import BASE_DIR, Settings
from features.state import DistractionReason

LOGGER = logging.getLogger(__name__)

NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263
LEFT_MOUTH_INDEX = 61
RIGHT_MOUTH_INDEX = 291
LEFT_EYE_INNER_INDEX = 133
RIGHT_EYE_INNER_INDEX = 362

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class DetectionSnapshot:
    """Detector output consumed by service and UI."""

    has_face: bool
    raw_reason: DistractionReason | None
    yaw_degrees: float = 0.0
    down_ratio: float = 0.0
    face_box: tuple[int, int, int, int] | None = None
    landmark_points: list[tuple[int, int]] = field(default_factory=list)
    debug_values: dict[str, str] = field(default_factory=dict)


class FocusDetector:
    """Extract face presence and simple attention heuristics from a frame."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend = "opencv"
        self.face_mesh = None

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

        self._initialize_mediapipe_if_possible()

    def analyze(self, frame: np.ndarray) -> DetectionSnapshot:
        """Analyze a frame and return a simple observation."""
        if self.face_mesh is not None:
            return self._analyze_with_mediapipe(frame)
        return self._analyze_with_opencv(frame)

    def close(self) -> None:
        """Release detector resources."""
        if self.face_mesh is not None:
            self.face_mesh.close()

    def _initialize_mediapipe_if_possible(self) -> None:
        model_path = self.settings.mediapipe_face_landmarker_path
        if not model_path:
            LOGGER.info(
                "MediaPipe model path not configured. Falling back to OpenCV face heuristics."
            )
            return

        resolved_model = (BASE_DIR / model_path).resolve()
        if not resolved_model.exists():
            LOGGER.warning(
                "MediaPipe model not found at %s. Falling back to OpenCV face heuristics.",
                resolved_model,
            )
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.face_landmarker import (
                FaceLandmarker,
                FaceLandmarkerOptions,
            )
        except Exception as error:  # pragma: no cover - defensive runtime path
            LOGGER.warning(
                "MediaPipe import failed (%s). Falling back to OpenCV heuristics.",
                error,
            )
            return

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_model)),
            num_faces=1,
        )
        self.face_mesh = FaceLandmarker.create_from_options(options)
        self.backend = "mediapipe"
        self._mp = mp
        LOGGER.info("Using MediaPipe face landmarker backend.")

    def _analyze_with_mediapipe(self, frame: np.ndarray) -> DetectionSnapshot:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self.face_mesh.detect(mp_image)

        if not result.face_landmarks:
            return self._no_face_snapshot()

        face_landmarks = result.face_landmarks[0]
        frame_height, frame_width = frame.shape[:2]
        landmark_points = self._normalized_to_pixel_points(
            face_landmarks, frame_width, frame_height
        )
        face_box = self._compute_face_box(landmark_points)
        yaw_degrees = self._estimate_yaw(face_landmarks, frame_width, frame_height)
        down_ratio = self._estimate_down_ratio(face_landmarks)
        raw_reason = self._reason_from_pose(yaw_degrees, down_ratio)

        return DetectionSnapshot(
            has_face=True,
            raw_reason=raw_reason,
            yaw_degrees=yaw_degrees,
            down_ratio=down_ratio,
            face_box=face_box,
            landmark_points=landmark_points,
            debug_values={
                "backend": self.backend,
                "face_detected": "yes",
                "raw_reason": raw_reason.value if raw_reason else "focused",
                "yaw_deg": f"{yaw_degrees:.1f}",
                "down_ratio": f"{down_ratio:.2f}",
            },
        )

    def _analyze_with_opencv(self, frame: np.ndarray) -> DetectionSnapshot:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        frontal_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(90, 90),
        )
        profile_faces = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(90, 90),
        )
        flipped_gray = cv2.flip(gray, 1)
        flipped_profile_faces = self.profile_cascade.detectMultiScale(
            flipped_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(90, 90),
        )
        mirrored_profile_faces = [
            (gray.shape[1] - x - w, y, w, h) for x, y, w, h in flipped_profile_faces
        ]

        all_profile_faces = list(profile_faces) + mirrored_profile_faces

        if len(frontal_faces) == 0 and len(all_profile_faces) == 0:
            return self._no_face_snapshot()

        if len(frontal_faces) == 0 and all_profile_faces:
            face_box = self._largest_box(all_profile_faces)
            landmark_points = self._box_landmarks(face_box)
            return DetectionSnapshot(
                has_face=True,
                raw_reason=DistractionReason.LOOKING_AWAY,
                yaw_degrees=self.settings.looking_away_yaw_threshold_degrees + 5.0,
                down_ratio=0.0,
                face_box=self._to_corner_box(face_box),
                landmark_points=landmark_points,
                debug_values={
                    "backend": self.backend,
                    "face_detected": "yes",
                    "raw_reason": DistractionReason.LOOKING_AWAY.value,
                    "yaw_deg": "profile",
                    "down_ratio": "0.00",
                },
            )

        face_box = self._largest_box(frontal_faces)
        x, y, width, height = face_box
        roi_gray = gray[y : y + height, x : x + width]
        upper_roi = roi_gray[: max(height // 2, 1), :]
        eyes = self.eye_cascade.detectMultiScale(
            upper_roi,
            scaleFactor=1.08,
            minNeighbors=8,
            minSize=(20, 20),
        )

        eye_centers: list[tuple[int, int]] = []
        for eye_x, eye_y, eye_w, eye_h in sorted(eyes, key=lambda item: item[0])[:2]:
            eye_centers.append((x + eye_x + (eye_w // 2), y + eye_y + (eye_h // 2)))

        yaw_degrees = 0.0
        down_ratio = 0.0
        raw_reason = None

        if all_profile_faces:
            raw_reason = DistractionReason.LOOKING_AWAY
            yaw_degrees = self.settings.looking_away_yaw_threshold_degrees + 2.0
        elif len(eye_centers) < 2:
            raw_reason = DistractionReason.LOOKING_DOWN
            down_ratio = self.settings.looking_down_ratio_threshold + 0.08
        else:
            face_center_x = x + (width / 2)
            eyes_center_x = sum(point[0] for point in eye_centers) / len(eye_centers)
            horizontal_offset = abs(eyes_center_x - face_center_x) / max(width, 1)
            eyes_center_y = sum(point[1] for point in eye_centers) / len(eye_centers)
            down_ratio = (eyes_center_y - y) / max(height, 1)
            yaw_degrees = horizontal_offset * 120.0

            if yaw_degrees >= self.settings.looking_away_yaw_threshold_degrees:
                raw_reason = DistractionReason.LOOKING_AWAY
            elif down_ratio >= self.settings.looking_down_ratio_threshold:
                raw_reason = DistractionReason.LOOKING_DOWN

        landmark_points = self._box_landmarks(face_box, eye_centers)
        return DetectionSnapshot(
            has_face=True,
            raw_reason=raw_reason,
            yaw_degrees=yaw_degrees,
            down_ratio=down_ratio,
            face_box=self._to_corner_box(face_box),
            landmark_points=landmark_points,
            debug_values={
                "backend": self.backend,
                "face_detected": "yes",
                "raw_reason": raw_reason.value if raw_reason else "focused",
                "yaw_deg": f"{yaw_degrees:.1f}" if isinstance(yaw_degrees, float) else str(yaw_degrees),
                "down_ratio": f"{down_ratio:.2f}",
            },
        )

    def _no_face_snapshot(self) -> DetectionSnapshot:
        return DetectionSnapshot(
            has_face=False,
            raw_reason=DistractionReason.NO_FACE,
            debug_values={
                "backend": self.backend,
                "face_detected": "no",
                "raw_reason": DistractionReason.NO_FACE.value,
                "yaw_deg": "0.0",
                "down_ratio": "0.00",
            },
        )

    def _reason_from_pose(
        self,
        yaw_degrees: float,
        down_ratio: float,
    ) -> DistractionReason | None:
        if abs(yaw_degrees) >= self.settings.looking_away_yaw_threshold_degrees:
            return DistractionReason.LOOKING_AWAY
        if down_ratio >= self.settings.looking_down_ratio_threshold:
            return DistractionReason.LOOKING_DOWN
        return None

    @staticmethod
    def _normalized_to_pixel_points(
        face_landmarks: list[object],
        frame_width: int,
        frame_height: int,
    ) -> list[tuple[int, int]]:
        points: list[tuple[int, int]] = []
        for landmark in face_landmarks:
            x = min(max(int(landmark.x * frame_width), 0), frame_width - 1)
            y = min(max(int(landmark.y * frame_height), 0), frame_height - 1)
            points.append((x, y))
        return points

    @staticmethod
    def _compute_face_box(
        landmark_points: list[tuple[int, int]],
    ) -> tuple[int, int, int, int] | None:
        if not landmark_points:
            return None

        xs = [point[0] for point in landmark_points]
        ys = [point[1] for point in landmark_points]
        return min(xs), min(ys), max(xs), max(ys)

    def _estimate_yaw(
        self,
        face_landmarks: list[object],
        frame_width: int,
        frame_height: int,
    ) -> float:
        image_points = np.array(
            [
                self._landmark_to_xy(face_landmarks, NOSE_TIP_INDEX, frame_width, frame_height),
                self._landmark_to_xy(face_landmarks, CHIN_INDEX, frame_width, frame_height),
                self._landmark_to_xy(
                    face_landmarks, LEFT_EYE_OUTER_INDEX, frame_width, frame_height
                ),
                self._landmark_to_xy(
                    face_landmarks, RIGHT_EYE_OUTER_INDEX, frame_width, frame_height
                ),
                self._landmark_to_xy(
                    face_landmarks, LEFT_MOUTH_INDEX, frame_width, frame_height
                ),
                self._landmark_to_xy(
                    face_landmarks, RIGHT_MOUTH_INDEX, frame_width, frame_height
                ),
            ],
            dtype=np.float64,
        )

        focal_length = frame_width
        camera_matrix = np.array(
            [
                [focal_length, 0, frame_width / 2],
                [0, focal_length, frame_height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _, _, _, _, _, _, euler_angles = cv2.RQDecomp3x3(rotation_matrix)
        return float(euler_angles[1])

    @staticmethod
    def _estimate_down_ratio(face_landmarks: list[object]) -> float:
        left_eye = face_landmarks[LEFT_EYE_INNER_INDEX]
        right_eye = face_landmarks[RIGHT_EYE_INNER_INDEX]
        nose = face_landmarks[NOSE_TIP_INDEX]
        chin = face_landmarks[CHIN_INDEX]

        eye_center_y = (left_eye.y + right_eye.y) / 2
        denominator = max(chin.y - eye_center_y, 1e-6)
        ratio = (nose.y - eye_center_y) / denominator
        return max(0.0, min(ratio, 1.5))

    @staticmethod
    def _landmark_to_xy(
        face_landmarks: list[object],
        index: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        landmark = face_landmarks[index]
        return landmark.x * frame_width, landmark.y * frame_height

    @staticmethod
    def _largest_box(boxes: list[tuple[int, int, int, int]] | np.ndarray) -> tuple[int, int, int, int]:
        return tuple(
            max(boxes, key=lambda item: int(item[2]) * int(item[3]))
        )

    @staticmethod
    def _to_corner_box(
        face_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        x, y, width, height = face_box
        return x, y, x + width, y + height

    @staticmethod
    def _box_landmarks(
        face_box: tuple[int, int, int, int],
        eye_centers: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        x, y, width, height = face_box
        points = [
            (x, y),
            (x + width, y),
            (x, y + height),
            (x + width, y + height),
            (x + (width // 2), y + (height // 2)),
        ]
        if eye_centers:
            points.extend(eye_centers)
        return points
