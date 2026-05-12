"""Clients utilities."""

from typing import Any

from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus
from nekomata.utils import get_utc_timestamp


def create_failed_response[ResponseT](
    response: ResponseT | None,
    fail_reason: str,
    created_at: float,
) -> ChatCompletionResponse[None]:
    """Create a failed chat completion object."""
    elapsed = get_utc_timestamp() - created_at
    return ChatCompletionResponse(
        created_at=created_at,
        elapsed=elapsed,
        status=ChatCompletionStatus.FAILED,
        original=response,
        fail_reason=fail_reason,
        content=None,
        finish_reason=None,
    )


def filter_none(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove fields with None value."""
    return {k: v for k, v in kwargs.items() if v is not None}
