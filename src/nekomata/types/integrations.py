"""Integrations."""

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field

from nekomata.utils.uuid import create_uuid


class ChatCompletionStatus(StrEnum):
    """Completion status enum."""

    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'


class ChatCompletionResponse[ResponseFormatT](BaseModel):
    """Chat completion response class."""

    id: Annotated[str, Field(description='Internal ID.')] = Field(default_factory=create_uuid)

    status: Annotated[
        ChatCompletionStatus,
        Field(description='Status of the chat completion call.'),
    ] = ChatCompletionStatus.SUCCESS
    fail_reason: Annotated[str | None, Field(description='Failed reason.')] = None

    original: Annotated[Any | None, Field(description='Original response object.', repr=False)] = None

    content: Annotated[str | None, Field(..., description='Generated content.')]

    reason: Annotated[str | None, Field(description='Generated reasoning content.')] = None

    finish_reason: Annotated[str | None, Field(..., description='Stop reason.')]

    parsed: Annotated[ResponseFormatT | None, Field(description='Parsed response.')] = None

    # Usage
    total_tokens: Annotated[int | None, Field(description='Total token usage.')] = None
    input_tokens: Annotated[int | None, Field(description='Input prompt token usage.')] = None
    output_tokens: Annotated[int | None, Field(description='Output token usage.')] = None

    cache_tokens: Annotated[int | None, Field(description='Cached tokens.')] = None
    reason_tokens: Annotated[int | None, Field(description='Reasoning tokens.')] = None
