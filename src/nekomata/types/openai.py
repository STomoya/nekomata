"""OpenAI specific types."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

from nekomata.types.clients import PackageSpecificArgs


class OpenAIChatCompletionCommonAttrs(NamedTuple):
    """OpenAI chat completion output classes' common attributes."""

    content: str | None
    reason: str | None
    finish_reason: str | None
    total_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    cache_tokens: int | None
    reason_tokens: int | None


@dataclass
class OpenAIArgs(PackageSpecificArgs):
    """OpenAI package specific API arguments."""

    api: Literal['chat_completions', 'responses'] = 'chat_completions'
