"""Anthropic Batch API Client Plugin."""

from typing import Any, Literal

from anthropic import AsyncAnthropic

from nekomata.clients.base import BatchAPIPlugin
from nekomata.clients.utils import filter_none, validate_and_expand_batch_args
from nekomata.utils import get_logger

logger = get_logger(__name__)


class AnthropicBatchAPIPlugin(BatchAPIPlugin):
    """Anthropic Batch API client plugin."""

    _client: AsyncAnthropic

    async def acreate_batch(
        self,
        *,
        model: str,
        prompt: str | list[str],
        system_prompt: str | list[str] | None = None,
        max_output_tokens: int | list[int] | None = None,
        response_format: type[Any] | list[type[Any]] | None = None,
        reasoning_effort: str | list[str] | None = None,
        custom_id: str | list[str] | None = None,
        mode: Literal['file', 'inline'] = 'inline',
    ) -> Any:
        """Create a new message batch.

        Args:
            model (str): The model name.
            prompt (str | list[str]): User prompt(s).
            system_prompt (str | list[str] | None): System prompt(s).
            max_output_tokens (int | list[int] | None): Maximum output tokens.
            response_format (type[Any] | list[type[Any]] | None): Response format(s).
            reasoning_effort (str | list[str] | None): Reasoning effort(s).
            custom_id (str | list[str] | None): Custom ID(s).
            mode (Literal['file', 'inline']): Batch mode (ignored for Anthropic).

        """
        if mode == 'file':
            msg = 'Anthropic only supports inline batch requests. Proceeding with "inline".'
            logger.warning(msg)

        expanded = validate_and_expand_batch_args(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            custom_id=custom_id,
        )

        logger.info(f"Creating Anthropic batch job with model '{model}' ({len(expanded)} requests)...")

        requests = []
        for item in expanded:
            messages = [{'role': 'user', 'content': item.prompt}]

            params: dict[str, Any] = {
                'model': model,
                'messages': messages,
                'max_tokens': item.max_output_tokens if item.max_output_tokens is not None else 4096,
            }

            if item.system_prompt is not None:
                params['system'] = item.system_prompt

            if item.reasoning_effort is not None:
                params['thinking'] = {'type': 'adaptive', 'display': 'summarized'}
                params['output_config'] = {'effort': item.reasoning_effort}

            if item.response_format is not None:
                params['output_format'] = item.response_format

            requests.append(
                {
                    'custom_id': item.custom_id,
                    'params': params,
                }
            )

        batch_job = await self._client.beta.messages.batches.create(
            requests=requests,
        )
        batch_job_id = getattr(batch_job, 'id', batch_job)
        logger.info(f"Successfully created Anthropic batch job '{batch_job_id}'.")
        return batch_job

    async def aretrieve_batch(self, batch_id: str, *, timeout: float | None = None) -> Any:
        """Retrieve the status and details of a batch job."""
        logger.debug(f"Retrieving Anthropic batch job '{batch_id}'...")
        kwargs = filter_none({'timeout': timeout})
        return await self._client.beta.messages.batches.retrieve(batch_id, **kwargs)

    async def acancel_batch(self, batch_id: str, *, timeout: float | None = None) -> Any:
        """Cancel an active batch job."""
        logger.info(f"Cancelling Anthropic batch job '{batch_id}'...")
        kwargs = filter_none({'timeout': timeout})
        return await self._client.beta.messages.batches.cancel(batch_id, **kwargs)

    async def alist_batches(
        self,
        *,
        after_id: str | None = None,
        before_id: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> Any:
        """List batch jobs."""
        logger.debug('Listing Anthropic batch jobs...')
        kwargs = filter_none(
            {
                'after_id': after_id,
                'before_id': before_id,
                'limit': limit,
                'timeout': timeout,
            }
        )
        return await self._client.beta.messages.batches.list(**kwargs)

    async def adelete_batch(self, batch_id: str, *, timeout: float | None = None) -> Any:
        """Delete a batch job."""
        logger.info(f"Deleting Anthropic batch job '{batch_id}'...")
        kwargs = filter_none({'timeout': timeout})
        return await self._client.beta.messages.batches.delete(batch_id, **kwargs)
