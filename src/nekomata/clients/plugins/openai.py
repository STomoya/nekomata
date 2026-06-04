"""OpenAI Batch API Client Plugin."""

import json
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from nekomata.clients.base import BatchAPIPlugin
from nekomata.clients.utils import filter_none, validate_and_expand_batch_args
from nekomata.utils import get_logger

logger = get_logger(__name__)


class OpenAIBatchAPIPlugin(BatchAPIPlugin):
    """OpenAI Batch API client plugin."""

    _client: AsyncOpenAI

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
        mode: Literal['file', 'inline'] = 'file',
    ) -> Any:
        """Create a new batch job.

        Args:
            model (str): The model name.
            prompt (str | list[str]): User prompt(s).
            system_prompt (str | list[str] | None): System prompt(s).
            max_output_tokens (int | list[int] | None): Maximum output tokens.
            response_format (type[Any] | list[type[Any]] | None): Response format(s).
            reasoning_effort (str | list[str] | None): Reasoning effort(s).
            custom_id (str | list[str] | None): Custom ID(s).
            mode (Literal['file', 'inline']): Batch mode (ignored for OpenAI).

        """
        expanded = validate_and_expand_batch_args(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            custom_id=custom_id,
        )

        logger.info(f"Creating OpenAI batch job with model '{model}' ({len(expanded)} requests)...")

        jsonl_lines = []
        for item in expanded:
            item_custom_id = item.custom_id

            messages = []
            if item.system_prompt is not None:
                messages.append({'role': 'system', 'content': item.system_prompt})
            messages.append({'role': 'user', 'content': item.prompt})

            body: dict[str, Any] = {
                'model': model,
                'messages': messages,
            }
            if item.max_output_tokens is not None:
                body['max_completion_tokens'] = item.max_output_tokens
            if item.reasoning_effort is not None:
                body['reasoning_effort'] = item.reasoning_effort
            if item.response_format is not None and issubclass(item.response_format, BaseModel):
                body['response_format'] = {
                    'type': 'json_schema',
                    'json_schema': {
                        'schema': to_strict_json_schema(item.response_format),
                        'name': item.response_format.__name__,
                        'strict': True,
                    },
                }

            req = {
                'custom_id': item_custom_id,
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': body,
            }
            jsonl_lines.append(json.dumps(req))

        file_content = '\n'.join(jsonl_lines) + '\n'

        logger.debug('Uploading batch file to OpenAI...')
        uploaded_file = await self._client.files.create(
            file=('batch.jsonl', file_content.encode('utf-8'), 'application/jsonl'),
            purpose='batch',
        )
        logger.info(f"Uploaded batch file to OpenAI with ID '{uploaded_file.id}'.")

        batch_job = await self._client.batches.create(
            completion_window='24h',
            endpoint='/v1/chat/completions',
            input_file_id=uploaded_file.id,
        )
        batch_job_id = getattr(batch_job, 'id', batch_job)
        logger.info(f"Successfully created OpenAI batch job '{batch_job_id}'.")
        return batch_job

    async def aretrieve_batch(self, batch_id: str, *, timeout: float | None = None) -> Any:
        """Retrieve the status and details of a batch job."""
        logger.debug(f"Retrieving OpenAI batch job '{batch_id}'...")
        kwargs = filter_none({'timeout': timeout})
        return await self._client.batches.retrieve(batch_id, **kwargs)

    async def acancel_batch(self, batch_id: str, *, timeout: float | None = None) -> Any:
        """Cancel an active batch job."""
        logger.info(f"Cancelling OpenAI batch job '{batch_id}'...")
        kwargs = filter_none({'timeout': timeout})
        return await self._client.batches.cancel(batch_id, **kwargs)

    async def alist_batches(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> Any:
        """List batch jobs."""
        logger.debug('Listing OpenAI batch jobs...')
        kwargs = filter_none(
            {
                'after': after,
                'limit': limit,
                'timeout': timeout,
            }
        )
        return await self._client.batches.list(**kwargs)

    async def adelete_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Delete a batch job (Not supported by OpenAI)."""
        logger.warning(f"Attempted to delete OpenAI batch job '{batch_id}' but it is not supported.")
        raise NotImplementedError('OpenAI does not support deleting batches.')
