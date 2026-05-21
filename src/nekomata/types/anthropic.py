"""Anthropic specific types."""

from typing import NamedTuple

from nekomata.types.clients import PackageSpecificArgs


class AnthropicMessagesCommonAttrs(NamedTuple):
    """Anthropic messages output classes' common attributes."""

    content: str | None
    reason: str | None
    finish_reason: str | None
    total_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    cache_tokens: int | None
    reason_tokens: int | None


class AnthropicArgs(PackageSpecificArgs):
    """Anthropic specific API arguments."""
