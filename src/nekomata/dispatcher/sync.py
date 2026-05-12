"""Sync dispatcher."""

import concurrent.futures
from contextlib import AbstractContextManager
from threading import Lock
from types import TracebackType
from typing import Any, Literal, Self, TypeVar, overload

from anyio.from_thread import BlockingPortal, start_blocking_portal

from nekomata.dispatcher.dispatcher import AsyncLLMDispatcher
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.utils import get_logger

ResponseFormatT = TypeVar('ResponseFormatT')

logger = get_logger(__name__)


class SyncLLMDispatcher:
    """Synchronous Adapter: Runs AsyncLLMDispatcher in a background thread.

    Provides a seamless synchronous API (returning Futures) without modifying
    the core asynchronous logic. This class is thread-safe and manages a
    dedicated background anyio portal thread.
    """

    def __init__(
        self,
        backend: str = 'asyncio',
        backend_options: dict[str, Any] | None = None,
    ) -> None:
        """Construct dispatcher."""
        self._async_dispatcher: AsyncLLMDispatcher = AsyncLLMDispatcher()

        self._backend: str = backend
        self._backend_options: dict[str, Any] | None = backend_options
        self._portal_cm: AbstractContextManager[BlockingPortal] | None = None
        self._portal: BlockingPortal | None = None
        self._leases: int = 0
        self._lock: Lock = Lock()

    def __enter__(self) -> Self:
        """Enter context, incrementing the reference count."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context, decrementing the reference count."""
        self.stop()

    def register_endpoint(self, *args: Any, **kwargs: Any) -> None:
        """Pass-through configuration to the underlying async dispatcher."""
        with self._lock:
            self._async_dispatcher.register_endpoint(*args, **kwargs)

    def start(self) -> None:
        """Increment the reference count and ensure the portal is running."""
        with self._lock:
            if self._portal_cm is None:
                self._portal_cm = start_blocking_portal(
                    backend=self._backend,
                    backend_options=self._backend_options,
                )
                self._portal = self._portal_cm.__enter__()
                logger.info('SyncLLMDispatcher instance started.')

            self._leases += 1

    def stop(self) -> None:
        """Decrement the reference count and release the portal if no longer used."""
        with self._lock:
            if self._portal_cm is None or self._portal is None or self._leases <= 0:
                raise RuntimeError('Dispatcher is not running. Call start() or use a context manager.')

            self._leases -= 1
            if self._leases == 0:
                try:
                    self._portal.call(self._async_dispatcher.close)
                except Exception:
                    logger.exception('Error closing async dispatcher during shutdown.')
                finally:
                    if self._portal_cm:
                        self._portal_cm.__exit__(None, None, None)
                    self._portal = None
                    self._portal_cm = None
                    logger.info('SyncLLMDispatcher instance stopped.')

    @overload
    def submit(
        self,
        endpoint_name: str,
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
        custom_id: str | None = None,
    ) -> concurrent.futures.Future[ChatCompletionResponse[None]]: ...

    @overload
    def submit(
        self,
        endpoint_name: str,
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
    ) -> concurrent.futures.Future[ChatCompletionResponse[ResponseFormatT]]: ...

    def submit(
        self,
        endpoint_name: str,
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
        custom_id: str | None = None,
    ) -> concurrent.futures.Future[ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]]:
        """Submit a request to the background loop, returning a synchronous Future."""
        with self._lock:
            if self._portal is None:
                raise RuntimeError('Dispatcher is not running. Call start() or use a context manager.')
            portal = self._portal

        logger.debug(f"Sync submission to '{endpoint_name}' (model: {model})")
        return portal.start_task_soon(
            self._async_dispatcher.submit,
            endpoint_name,
            model,
            prompt,
            system_prompt,
            max_output_tokens,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            seed,
            response_format,
            reasoning_effort,
            extra_body,
            custom_id,
        )
