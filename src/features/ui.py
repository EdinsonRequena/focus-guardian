"""Overlay rendering for the live webcam demo."""

import cv2
import numpy as np

from core.settings import Settings
from features.detector import DetectionSnapshot
from features.state import FocusState, SessionSnapshot

GREEN = (46, 204, 113)
AMBER = (0, 160, 255)
WHITE = (245, 245, 245)
PANEL = (20, 20, 20)


class FocusUI:
    """Draw the live overlay over the camera feed."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.window_title = settings.window_title

    def render(
        self,
        frame: np.ndarray,
        session: SessionSnapshot,
        detection: DetectionSnapshot,
    ) -> np.ndarray:
        """Return a frame with the full MVP overlay applied."""
        color = GREEN if session.current_state == FocusState.FOCUSED else AMBER
        output = frame.copy()

        self._draw_frame_border(output, color)
        self._draw_status_panel(output, session, color)
        self._draw_counter_panel(output, session, color)

        if detection.face_box is not None:
            self._draw_face_box(output, detection.face_box, color)

        if self.settings.show_landmarks and detection.landmark_points:
            self._draw_landmarks(
                output,
                detection.landmark_points,
                detection.landmark_connections,
            )

        if self.settings.show_debug_metrics:
            self._draw_debug_panel(output, session, detection)

        return output

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        session: SessionSnapshot,
        color: tuple[int, int, int],
    ) -> None:
        self._draw_panel(frame, (20, 20), (360, 125))
        cv2.putText(
            frame,
            f"STATUS: {session.current_state.value.upper()}",
            (36, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )

        reason_text = (
            session.current_reason.value if session.current_reason is not None else "none"
        )
        cv2.putText(
            frame,
            f"reason: {reason_text}",
            (36, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            WHITE,
            2,
            cv2.LINE_AA,
        )

    def _draw_counter_panel(
        self,
        frame: np.ndarray,
        session: SessionSnapshot,
        color: tuple[int, int, int],
    ) -> None:
        panel_width = 220
        x1 = frame.shape[1] - panel_width - 20
        self._draw_panel(frame, (x1, 20), (panel_width, 95))
        cv2.putText(
            frame,
            "DISTRACTIONS",
            (x1 + 18, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(session.distraction_count),
            (x1 + 18, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            color,
            3,
            cv2.LINE_AA,
        )

    def _draw_debug_panel(
        self,
        frame: np.ndarray,
        session: SessionSnapshot,
        detection: DetectionSnapshot,
    ) -> None:
        lines = [
            f"backend: {detection.debug_values.get('backend', 'unknown')}",
            f"face: {detection.debug_values.get('face_detected', 'no')}",
            f"candidate: {session.candidate_reason.value if session.candidate_reason else 'none'}",
            f"turn_ratio: {detection.debug_values.get('turn_ratio', '0.000')}",
            f"head_down: {detection.debug_values.get('head_down_ratio', '0.000')}",
            f"eyes_down: {detection.debug_values.get('eyes_down_ratio', '0.000')}",
            f"baseline: {detection.debug_values.get('baseline_ready', 'no')}",
            f"eyes_measure: {detection.debug_values.get('eyes_measure', 'n/a')}",
            f"eyes_base: {detection.debug_values.get('eyes_baseline', 'n/a')}",
            f"eyes_delta: {detection.debug_values.get('eyes_delta', 'n/a')}",
            f"down_ratio: {detection.debug_values.get('down_ratio', '0.00')}",
            f"analysis: {detection.debug_values.get('analysis', 'n/a')}",
        ]
        height = 32 + (len(lines) * 26)
        y1 = frame.shape[0] - height - 20
        self._draw_panel(frame, (20, y1), (310, height))

        baseline = y1 + 32
        for line in lines:
            cv2.putText(
                frame,
                line,
                (36, baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                WHITE,
                1,
                cv2.LINE_AA,
            )
            baseline += 24

    @staticmethod
    def _draw_face_box(
        frame: np.ndarray,
        face_box: tuple[int, int, int, int],
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_landmarks(
        frame: np.ndarray,
        landmark_points: list[tuple[int, int]],
        landmark_connections: list[tuple[int, int]],
    ) -> None:
        line_color = (255, 215, 0)
        point_color = (0, 255, 255)

        for start, end in landmark_connections:
            if start >= len(landmark_points) or end >= len(landmark_points):
                continue

            cv2.line(
                frame,
                landmark_points[start],
                landmark_points[end],
                line_color,
                1,
                cv2.LINE_AA,
            )

        for x, y in landmark_points:
            cv2.circle(frame, (x, y), 1, point_color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_frame_border(
        frame: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        cv2.rectangle(
            frame,
            (8, 8),
            (frame.shape[1] - 8, frame.shape[0] - 8),
            color,
            4,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_panel(
        frame: np.ndarray,
        top_left: tuple[int, int],
        size: tuple[int, int],
        alpha: float = 0.6,
    ) -> None:
        x1, y1 = top_left
        width, height = size
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x1 + width, y1 + height),
            PANEL,
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
