"""Plugin related types."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BatchRequestItem:
    """Class representing a single item in a batch request."""

    prompt: str
    custom_id: str
    system_prompt: str | None = None
    max_output_tokens: int | None = None
    response_format: Any = None
    reasoning_effort: str | None = None
