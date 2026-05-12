"""ABC for LLM clients."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Literal, TypeVar, overload

import anyio
import httpx

from nekomata.types.integrations import ChatCompletionResponse
from nekomata.utils import get_logger

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

    @abstractmethod
    def convert_output(
        self, response: ResponseT, created_at: float
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Convert raw response to a flatter response object."""
        raise NotImplementedError

    # Overload for regular LLM api calls.
    @overload
    @abstractmethod
    async def acompletion(
        self,
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
        response_format: None = None,
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResponse[None]: ...

    # Overload for structured output LLM api calls.
    @overload
    @abstractmethod
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
    ) -> ChatCompletionResponse[ResponseFormatT]: ...

    @abstractmethod
    async def acompletion(
        self,
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
        response_format: type[ResponseFormatT] | None = None,
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Async completion API call."""
        raise NotImplementedError
