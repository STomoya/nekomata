"""Public helper utilities for LLM clients."""

from typing import Any

import anyio

from nekomata.clients.plugins.anthropic import AnthropicBatchAPIPlugin
from nekomata.clients.plugins.google import GoogleBatchAPIPlugin
from nekomata.clients.plugins.openai import OpenAIBatchAPIPlugin
from nekomata.utils import get_logger

logger = get_logger(__name__)


def _extract_status_value(
    status_obj: Any,
    *,
    is_openai: bool,
    is_anthropic: bool,
    is_google: bool,
) -> Any:
    """Extract status value from the retrieved status object."""
    if is_openai:
        return getattr(status_obj, 'status', None)
    if is_anthropic:
        return getattr(status_obj, 'processing_status', None)
    if is_google:
        return getattr(status_obj, 'state', None)

    if hasattr(status_obj, 'status'):
        return status_obj.status
    if hasattr(status_obj, 'state'):
        return status_obj.state
    if hasattr(status_obj, 'processing_status'):
        return status_obj.processing_status
    if isinstance(status_obj, dict):
        return status_obj.get('status') or status_obj.get('state') or status_obj.get('processing_status')
    return None


async def apoll_batch(
    client: Any,
    batch_id: str,
    *,
    stop_statuses: list[str] | None = None,
    interval: float = 10.0,
) -> Any:
    """Poll a batch job until it reaches a terminal status or completed state.

    For officially supported clients, it waits for finish-related batch status automatically.
    Otherwise, stop_statuses must be provided to terminate polling.

    Args:
        client (Any): The LLM client.
        batch_id (str): The ID of the batch job to poll.
        stop_statuses (list[str] | None, optional): Statuses to stop polling. Defaults to None.
        interval (float, optional): The polling interval in seconds. Defaults to 10.0.

    Returns:
        Any: The final batch job status object.

    """
    is_openai = isinstance(client, OpenAIBatchAPIPlugin)
    is_anthropic = isinstance(client, AnthropicBatchAPIPlugin)
    is_google = isinstance(client, GoogleBatchAPIPlugin)

    if not (is_openai or is_anthropic or is_google) and stop_statuses is None:
        raise ValueError('stop_statuses must be provided for non-officially supported clients.')

    logger.info(f"Starting batch job polling for '{batch_id}' (interval={interval}s)...")

    while True:
        status_obj = await client.aretrieve_batch(batch_id)

        # Extract status value
        status_val = _extract_status_value(
            status_obj,
            is_openai=is_openai,
            is_anthropic=is_anthropic,
            is_google=is_google,
        )

        if status_val is None:
            if isinstance(status_obj, str):
                status_val = status_obj
            else:
                raise ValueError(f'Could not extract status from retrieve_batch result: {status_obj}')

        # Convert status value to string representation
        if hasattr(status_val, 'name'):
            status_str = status_val.name
        elif hasattr(status_val, 'value'):
            status_str = str(status_val.value)
        else:
            status_str = str(status_val)

        logger.debug(f"Polled batch job '{batch_id}': status='{status_str}'")

        # Check for stop status
        if is_openai:
            if status_str.lower() in ('completed', 'failed', 'expired', 'cancelled'):
                logger.info(f"Batch job '{batch_id}' reached terminal status '{status_str}'. Stop polling.")
                return status_obj
        elif is_anthropic:
            if status_str.lower() == 'ended':
                logger.info(f"Batch job '{batch_id}' reached terminal status '{status_str}'. Stop polling.")
                return status_obj
        elif is_google:
            if getattr(status_obj, 'done', False) is True:
                logger.info(f"Batch job '{batch_id}' completed (done=True). Stop polling.")
                return status_obj
            if any(term in status_str.upper() for term in ('SUCCEEDED', 'FAILED', 'CANCELLED', 'EXPIRED')):
                logger.info(f"Batch job '{batch_id}' reached terminal status '{status_str}'. Stop polling.")
                return status_obj
        else:
            # Custom client logic using stop_statuses
            assert stop_statuses is not None
            if status_str in stop_statuses or status_str.lower() in [s.lower() for s in stop_statuses]:
                logger.info(f"Batch job '{batch_id}' reached stop status '{status_str}'. Stop polling.")
                return status_obj

        await anyio.sleep(interval)
