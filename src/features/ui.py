"""UI placeholders for the first project scaffold."""

import logging

from features.state import FocusStatus

LOGGER = logging.getLogger(__name__)


class FocusUI:
    """Placeholder local UI boundary for the future overlay/window."""

    def __init__(self, window_title: str) -> None:
        self.window_title = window_title

    def render(self, status: FocusStatus) -> None:
        """Emit a debug line until a real UI exists."""
        LOGGER.debug(
            "UI placeholder | window=%s | state=%s | reason=%s",
            self.window_title,
            status.state.value,
            status.reason,
        )
