"""Typed error for the workflow's discriminated result contract.

run_workflow returns a discriminated dict: {"status": "ok", "results": [...]}
on success, {"status": "error", "error": {"type", "message"}} on failure.
SearchError is the internal carrier for that failure shape. run_workflow catches
it and returns its dict rendering, so it never leaks a bare string. In the
LangGraph tool path a raised SearchError is caught by ToolNode and serialized
into a ToolMessage instead of crashing the host process.
"""

from __future__ import annotations

from typing import Any


class SearchError(Exception):
    """Workflow failure carrying a machine-readable type and a human message."""

    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        """Render the discriminated error envelope that run_workflow returns."""
        return {
            "status": "error",
            "error": {"type": self.error_type, "message": self.message},
        }
