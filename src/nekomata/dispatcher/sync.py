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
        max_model_retry: int = 1,
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
        max_model_retry: int = 1,
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
        max_model_retry: int = 1,
    ) -> concurrent.futures.Future[ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]]:
        """Execute an LLM request.

        Args:
            endpoint_name (str): The name of the endpoint to send the request to.
            model (str): Model name.
            prompt (str): The prompt to send.
            response_format (type[ResponseFormatT] | None, optional): Response format defined as a pydantic BaseModel
                subclass. We currently do not support any other formats. Defaults to None.
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

        Returns:
            ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]: Response from the API.

        """
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
            response_format,
            system_prompt,
            max_output_tokens,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            seed,
            reasoning_effort,
            extra_body,
            custom_id,
            max_model_retry,
        )
