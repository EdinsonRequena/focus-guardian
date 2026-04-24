"""State transitions and distraction episode management."""

from core.settings import Settings
from features.detector import DetectionSnapshot
from features.state import (
    DistractionReason,
    FocusState,
    SessionSnapshot,
    SessionState,
)


class FocusGuardianService:
    """Apply timing rules and expose a stable session snapshot."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = SessionState()

    def update(self, observation: DetectionSnapshot, now: float) -> SessionSnapshot:
        """Update the session from the latest detector observation."""
        self._update_candidate(observation.raw_reason, now)
        valid_reason = self._get_valid_reason(now)

        should_play_audio = False
        audio_reason: DistractionReason | None = None

        if self.state.current_state == FocusState.DISTRACTED:
            if valid_reason is not None:
                self.state.current_reason = valid_reason
                self.state.active_episode_reason = valid_reason

            if self._ready_to_recover(observation.raw_reason, now):
                self.state.current_state = FocusState.FOCUSED
                self.state.current_reason = None
                self.state.stable_since = None
                self.state.active_episode_reason = None
            else:
                self.state.current_reason = self.state.active_episode_reason
        else:
            if valid_reason is not None:
                self.state.current_state = FocusState.DISTRACTED
                self.state.current_reason = valid_reason
                self.state.active_episode_reason = valid_reason
                self.state.stable_since = None
                self.state.distraction_count += 1
                if self._should_play_audio(now):
                    should_play_audio = True
                    audio_reason = valid_reason
                    self.state.last_audio_at = now

        if observation.raw_reason is None:
            if self.state.stable_since is None:
                self.state.stable_since = now
        else:
            self.state.stable_since = None

        debug_values = dict(observation.debug_values)
        debug_values["candidate_reason"] = (
            self.state.candidate_reason.value if self.state.candidate_reason else "none"
        )
        debug_values["status"] = self.state.current_state.value

        return SessionSnapshot(
            current_state=self.state.current_state,
            current_reason=self.state.current_reason,
            distraction_count=self.state.distraction_count,
            should_play_audio=should_play_audio,
            audio_reason=audio_reason,
            candidate_reason=self.state.candidate_reason,
            debug_values=debug_values,
        )

    def reset_distraction_count(self) -> None:
        """Reset the session distraction counter for a fresh demo pass."""
        self.state.distraction_count = 0

    def _update_candidate(
        self, raw_reason: DistractionReason | None, now: float
    ) -> None:
        if raw_reason != self.state.candidate_reason:
            self.state.candidate_reason = raw_reason
            self.state.candidate_started_at = now if raw_reason is not None else None

    def _get_valid_reason(self, now: float) -> DistractionReason | None:
        candidate_reason = self.state.candidate_reason
        if candidate_reason is None or self.state.candidate_started_at is None:
            return None

        elapsed = now - self.state.candidate_started_at
        threshold = self._threshold_for(candidate_reason)
        if elapsed >= threshold:
            return candidate_reason

        return None

    def _ready_to_recover(
        self, raw_reason: DistractionReason | None, now: float
    ) -> bool:
        if raw_reason is not None:
            return False

        if self.state.stable_since is None:
            self.state.stable_since = now
            return False

        return (now - self.state.stable_since) >= self.settings.recovery_threshold_seconds

    def _threshold_for(self, reason: DistractionReason) -> float:
        if reason == DistractionReason.NO_FACE:
            return self.settings.no_face_threshold_seconds
        if reason == DistractionReason.LOOKING_AWAY:
            return self.settings.looking_away_threshold_seconds
        return self.settings.looking_down_threshold_seconds

    def _should_play_audio(self, now: float) -> bool:
        if self.state.last_audio_at is None:
            return True

        return (now - self.state.last_audio_at) >= self.settings.audio_cooldown_seconds
