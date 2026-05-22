"""OpenAI specific types."""

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


class ResponsesArgs(NamedTuple):
    """Addtional arguments for the responses API not supported by the chat completion endpoint."""

    response_id: str | None = None
    store: bool = True


class OpenAIArgs(PackageSpecificArgs):
    """OpenAI package specific API arguments."""

    api: Literal['chat_completions', 'responses'] = 'chat_completions'
    responses_args: ResponsesArgs | None = None
