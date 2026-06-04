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
    list_args = {}
    single_args = {}
    for name, val in [
        ('prompt', prompt),
        ('system_prompt', system_prompt),
        ('max_output_tokens', max_output_tokens),
        ('response_format', response_format),
        ('reasoning_effort', reasoning_effort),
        ('custom_id', custom_id),
    ]:
        if isinstance(val, list):
            list_args[name] = val
        else:
            single_args[name] = val

    if not list_args:
        raise ValueError(
            'At least one of prompt, system_prompt, max_output_tokens, response_format, '
            'reasoning_effort, or custom_id must be a list.'
        )

    lengths = {name: len(val) for name, val in list_args.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        details = ', '.join(f'{name}: {length}' for name, length in lengths.items())
        raise ValueError(f'Lengths of list arguments do not match: {details}')

    batch_len = next(iter(unique_lengths))
    expanded = []

    for i in range(batch_len):
        item_kwargs = {}
        for name in ['prompt', 'system_prompt', 'max_output_tokens', 'response_format', 'reasoning_effort']:
            if name in list_args:
                item_kwargs[name] = list_args[name][i]
            else:
                item_kwargs[name] = single_args[name]

        # Handle custom_id specifically to guarantee uniqueness
        if 'custom_id' in list_args:
            item_kwargs['custom_id'] = list_args['custom_id'][i]
        elif single_args['custom_id'] is not None:
            item_kwargs['custom_id'] = f'{single_args["custom_id"]}-{i}'
        else:
            item_kwargs['custom_id'] = f'req-{create_uuid()}'

        expanded.append(BatchRequestItem(**item_kwargs))
    return expanded
