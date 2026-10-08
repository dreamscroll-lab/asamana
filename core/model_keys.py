"""Model keys: a deployment holds every key its config uses, or none of them.

Without keys the backend still serves everything that reads stored worlds (list, replay, map,
traces); whatever would call a model is refused up front by the API.
"""

from __future__ import annotations


class ModelKeysMissing(RuntimeError):
    """A model call was attempted in a deployment started without model keys."""

    def __init__(self) -> None:
        super().__init__(
            "Model API keys are not configured. Run ./deploy/asamana.sh keys to add them."
        )
