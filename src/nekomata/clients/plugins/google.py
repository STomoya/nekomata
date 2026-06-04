"""Google GenAI Batch API Client Plugin."""

import io
import json
from typing import Any, Literal

from google.genai import Client, types
from pydantic import BaseModel

from nekomata.clients.base import BatchAPIPlugin
from nekomata.clients.utils import validate_and_expand_batch_args
from nekomata.utils import get_logger

logger = get_logger(__name__)


class GoogleBatchAPIPlugin(BatchAPIPlugin):
    """Google GenAI Batch API client plugin."""

    _client: Client

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
            mode (Literal['file', 'inline']): Batch mode.

        """
        expanded = validate_and_expand_batch_args(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            custom_id=custom_id,
        )

        logger.info(f"Creating Google batch job with model '{model}' ({len(expanded)} requests, mode '{mode}')...")

        inlined_requests = []
        for item in expanded:
            thinking_config = None
            if item.reasoning_effort is not None:
                thinking_config = types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=item.reasoning_effort,  # ty: ignore[invalid-argument-type]
                )

            config = types.GenerateContentConfig(
                system_instruction=item.system_prompt,
                max_output_tokens=item.max_output_tokens,
                response_schema=item.response_format,
                response_mime_type='application/json' if item.response_format else 'text/plain',
                thinking_config=thinking_config,
            )

            inlined_requests.append(
                types.InlinedRequest(
                    model=model,
                    contents=item.prompt,
                    config=config,
                    metadata={'custom_id': item.custom_id},
                )
            )

        if mode == 'inline':
            batch_job = await self._client.aio.batches.create(
                model=model,
                src=inlined_requests,
            )
            batch_job_name = getattr(batch_job, 'name', batch_job)
            logger.info(f"Successfully created Google batch job '{batch_job_name}'.")
            return batch_job

        # File mode (default): format each request as `{"key": custom_id, "request": InlinedRequest}`
        # and upload via File API
        jsonl_lines = []
        for idx, req in enumerate(inlined_requests):
            req_dict = req.model_dump(by_alias=True, exclude_none=True)
            if 'config' in req_dict and req_dict['config'] and 'responseSchema' in req_dict['config']:
                schema_val = req_dict['config']['responseSchema']
                if isinstance(schema_val, type) and issubclass(schema_val, BaseModel):
                    req_dict['config']['responseSchema'] = schema_val.model_json_schema()
            jsonl_lines.append(json.dumps({'key': expanded[idx].custom_id, 'request': req_dict}))

        file_content = '\n'.join(jsonl_lines) + '\n'
        file_obj = io.BytesIO(file_content.encode('utf-8'))
        upload_config = types.UploadFileConfig(mime_type='application/jsonl')
        logger.debug('Uploading batch file to Google...')
        uploaded_file = await self._client.aio.files.upload(file=file_obj, config=upload_config)
        logger.info(f"Uploaded batch file to Google with name '{uploaded_file.name}'.")

        src = types.BatchJobSource(
            file_name=uploaded_file.name,
        )

        batch_job = await self._client.aio.batches.create(
            model=model,
            src=src,
        )
        batch_job_name = getattr(batch_job, 'name', batch_job)
        logger.info(f"Successfully created Google batch job '{batch_job_name}'.")
        return batch_job

    async def aretrieve_batch(self, batch_id: str, *, config: Any = None) -> Any:
        """Retrieve the status and details of a batch job."""
        logger.debug(f"Retrieving Google batch job '{batch_id}'...")
        return await self._client.aio.batches.get(name=batch_id, config=config)

    async def acancel_batch(self, batch_id: str, *, config: Any = None) -> Any:
        """Cancel an active batch job."""
        logger.info(f"Cancelling Google batch job '{batch_id}'...")
        return await self._client.aio.batches.cancel(name=batch_id, config=config)

    async def alist_batches(self, *, config: Any = None) -> Any:
        """List batch jobs."""
        logger.debug('Listing Google batch jobs...')
        return await self._client.aio.batches.list(config=config)

    async def adelete_batch(self, batch_id: str, *, config: Any = None) -> Any:
        """Delete a batch job."""
        logger.info(f"Deleting Google batch job '{batch_id}'...")
        return await self._client.aio.batches.delete(name=batch_id, config=config)
