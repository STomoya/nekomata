"""ABC for LLM clients."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from json import JSONDecodeError
from typing import Any, Literal, TypeVar, overload

import anyio
import httpx
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from nekomata.clients.utils import create_failed_response
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.utils import get_logger
from nekomata.utils.misc import get_utc_timestamp

ResponseT = TypeVar('ResponseT')
ResponseFormatT = TypeVar('ResponseFormatT')

logger = get_logger(__name__)


class ClientABC(ABC):
    """Abstract class for LLM clients."""

    def __init__(
        self,
        max_concurrent: int | None = None,
        max_connections: int | None = 100,
        max_keepalive: int | None = 10,
        keepalive_expiry: float | None = None,
        timeout: float | None = 60.0,
        ssl_verify: str | bool = True,
    ) -> None:
        """Construct client.

        Args:
            max_concurrent (int | None): Maximum concurrent requests. Defaults to None.
            max_connections (int, optional): Maximum connections per connection pool. Defaults to 100.
            max_keepalive (int, optional): Maximum keep alive connections. Defaults to 10.
            keepalive_expiry (float | None, optional): Keep alive expiration time. Defaults to None.
            timeout (float, optional): Timeout for the client. Defaults to 60.0.
            ssl_verify (str | bool, optional): SSL verification specifications. Defaults to True.

        """
        _limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )

        _timeout = (
            httpx.Timeout(timeout=timeout)
            if isinstance(timeout, float) and timeout > 0
            else httpx.Timeout(timeout=None)
        )

        self._http_client = httpx.AsyncClient(
            limits=_limits,
            timeout=_timeout,
            verify=ssl_verify,
        )

        self._semaphore = anyio.Semaphore(initial_value=max_concurrent) if max_concurrent else nullcontext()

        self._initialized = False

    @property
    def initialized(self) -> bool:
        """Is client initialized."""
        return self._initialized

    @property
    def semaphore(self) -> anyio.Semaphore | nullcontext:
        """Semaphore."""
        return self._semaphore

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        logger.debug('Closing underlying HTTPX client.')
        await self._http_client.aclose()

    # region abstractmethod

    @abstractmethod
    def convert_output(
        self, response: ResponseT, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Convert raw response to a flatter response object."""
        raise NotImplementedError

    @abstractmethod
    async def _acompletion(
        self,
        created_at: float,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        response_format: type[ResponseFormatT] | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Actual async API call implementation."""
        raise NotImplementedError

    # endregion

    def handle_exception(
        self, err_msg: str, exc: Exception, created_at: float, custom_id: str | None
    ) -> ChatCompletionResponse[None]:
        """Log exception."""
        logger.exception(err_msg)
        return create_failed_response(response=None, fail_reason=f'{exc!s}', created_at=created_at, custom_id=custom_id)

    # Overload for regular LLM api calls.
    @overload
    async def acompletion(
        self,
        model: str,
        prompt: str,
        response_format: None = None,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
        max_model_retry: int = 1,
    ) -> ChatCompletionResponse[None]: ...

    # Overload for structured output LLM api calls.
    @overload
    async def acompletion(
        self,
        model: str,
        prompt: str,
        response_format: type[ResponseFormatT],
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
        max_model_retry: int = 1,
    ) -> ChatCompletionResponse[ResponseFormatT]: ...

    async def acompletion(
        self,
        model: str,
        prompt: str,
        response_format: type[ResponseFormatT] | None = None,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        seed: int | None = None,
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
        max_model_retry: int = 1,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Async completion API call.

        Args:
            model (str): The name of the model. Example: `gpt-5`.
            prompt (str): User input.
            system_prompt (str | None, optional): System prompt. Defaults to None.
            max_output_tokens (str | None, optional): Maximum output tokens. Defaults to None.
            temperature (float | None, optional): [Sampling] Temperature parameter. Defaults to None.
            top_p (float | None, optional): [Sampling] Top-P parameter. Defaults to None.
            top_k (int | None, optional): [Sampling] Top-K parameter. Defaults to None.
            presence_penalty (float | None, optional): [Sampling] Presence penalty. Defaults to None.
            frequency_penalty (float | None, optional): [Sampling] Frequency penalty. Defatuls to None.
            seed (int | None): [Sampling] Random seed. Defaults to None.
            response_format (type[BaseModel] | None, optional): JSON response format defined as a pydantic model.
                Defaults to None.
            reasoning_effort (Literal['high', 'medium', 'low', 'minimal'] | None, optional): Reasoning effort.
                Defaults to None.
            extra_body (dict[str, Any] | None, optional): Extra body.
            custom_id (str | None, optional): Custom ID. This value will overwrite the response object's ID field.
                Defaults to None.
            max_model_retry (int, optional): Maximum number of retries when failed to validate generated content to
                pydantic model. Defaults to 1.

        """
        created_at = get_utc_timestamp()

        logger.debug(f'Entering semaphore for model: {model}')
        async with self.semaphore:
            logger.debug(f'Acquired semaphore for model: {model}')
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_attempt_number=max_model_retry),
                    wait=wait_fixed(0.1),
                    # Only retry on pydantic model validation errors.
                    retry=retry_if_exception_type((ValidationError, JSONDecodeError)),
                    reraise=True,
                ):
                    with attempt:
                        response = await self._acompletion(
                            created_at=created_at,
                            model=model,
                            prompt=prompt,
                            response_format=response_format,
                            system_prompt=system_prompt,
                            max_output_tokens=max_output_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            seed=seed,
                            reasoning_effort=reasoning_effort,
                            extra_body=extra_body,
                            custom_id=custom_id,
                        )
                    if (
                        hasattr(attempt.retry_state.outcome, 'failed')
                        and attempt.retry_state.outcome.failed
                        and isinstance(attempt.retry_state.outcome.exception(), (ValidationError, JSONDecodeError))
                    ):
                        err_msg = 'Failed on validating response content to the provided pydantic model.'
                        if max_model_retry > 1:
                            err_msg += f' [Attempt {attempt.retry_state.attempt_number} / {max_model_retry}]'
                        logger.warning(err_msg)
            except Exception as e:
                return self.handle_exception('API error', exc=e, created_at=created_at, custom_id=custom_id)
            else:
                return response
