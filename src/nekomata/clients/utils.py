"""Clients utilities."""

from typing import Any

from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus
from nekomata.types.plugins import BatchRequestItem
from nekomata.utils import get_utc_timestamp
from nekomata.utils.uuid import create_uuid


def create_failed_response[ResponseT](
    response: ResponseT | None,
    fail_reason: str,
    created_at: float,
    custom_id: str | None = None,
) -> ChatCompletionResponse[None]:
    """Create a failed chat completion object."""
    id = custom_id or create_uuid()
    elapsed = get_utc_timestamp() - created_at
    return ChatCompletionResponse(
        id=id,
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


def _expand_arg[T](arg: T | list[T], length: int) -> list[T]:
    """Expand a value or list to a given length."""
    if isinstance(arg, list):
        return arg
    output = [arg] * length
    return output


def validate_and_expand_batch_args(
    prompt: str | list[str],
    system_prompt: str | list[str] | None,
    max_output_tokens: int | list[int] | None,
    response_format: Any,
    reasoning_effort: str | list[str] | None,
    custom_id: str | list[str] | None = None,
) -> list[BatchRequestItem]:
    """Validate and expand batch arguments.

    At least one of prompt, system_prompt, max_output_tokens, response_format,
    reasoning_effort, or custom_id must be a list. If multiple arguments are lists,
    their lengths must match. Any single values will be duplicated to match.

    Args:
        prompt (str | list[str]): User prompt(s).
        system_prompt (str | list[str] | None): System prompt(s).
        max_output_tokens (int | list[int] | None): Max output token(s).
        response_format (Any): Response format(s).
        reasoning_effort (str | list[str] | None): Reasoning effort(s).
        custom_id (str | list[str] | None): Custom ID(s).

    Returns:
        list[BatchRequestItem]: A list of BatchRequestItem, one for each request in the batch.

    """
    list_lengths = {}
    if isinstance(prompt, list):
        list_lengths['prompt'] = len(prompt)
    if isinstance(system_prompt, list):
        list_lengths['system_prompt'] = len(system_prompt)
    if isinstance(max_output_tokens, list):
        list_lengths['max_output_tokens'] = len(max_output_tokens)
    if isinstance(response_format, list):
        list_lengths['response_format'] = len(response_format)
    if isinstance(reasoning_effort, list):
        list_lengths['reasoning_effort'] = len(reasoning_effort)
    if isinstance(custom_id, list):
        list_lengths['custom_id'] = len(custom_id)

    if not list_lengths:
        raise ValueError(
            'At least one of prompt, system_prompt, max_output_tokens, response_format, '
            'reasoning_effort, or custom_id must be a list.'
        )

    unique_lengths = set(list_lengths.values())
    if len(unique_lengths) > 1:
        details = ', '.join(f'{name}: {length}' for name, length in list_lengths.items())
        raise ValueError(f'Lengths of list arguments do not match: {details}')

    batch_len = next(iter(unique_lengths))

    prompts = _expand_arg(prompt, batch_len)
    system_prompts = _expand_arg(system_prompt, batch_len)
    max_tokens_list = _expand_arg(max_output_tokens, batch_len)
    formats = _expand_arg(response_format, batch_len)
    reasoning_efforts = _expand_arg(reasoning_effort, batch_len)

    custom_ids: list[str]
    if isinstance(custom_id, list):
        custom_ids = custom_id
    elif custom_id is not None:
        custom_ids = [f'{custom_id}-{i}' for i in range(batch_len)]
    else:
        custom_ids = [f'req-{create_uuid()}' for _ in range(batch_len)]

    expanded = []
    for i in range(batch_len):
        expanded.append(
            BatchRequestItem(
                prompt=prompts[i],
                custom_id=custom_ids[i],
                system_prompt=system_prompts[i],
                max_output_tokens=max_tokens_list[i],
                response_format=formats[i],
                reasoning_effort=reasoning_efforts[i],
            )
        )
    return expanded
