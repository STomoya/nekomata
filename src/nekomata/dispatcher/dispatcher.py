"""Dispatcher."""

from types import TracebackType
from typing import Any, Literal, Self, TypeVar, overload

import anyio

from nekomata.clients.factory import create_client
from nekomata.clients.providers.anthropic import AnthropicClient
from nekomata.clients.providers.google import GoogleClient
from nekomata.clients.providers.openai import OpenAIClient
from nekomata.types.dispatcher import EndpointConfig
from nekomata.types.integrations import ChatCompletionResponse, ChatCompletionStatus
from nekomata.utils import get_logger

ResponseFormatT = TypeVar('ResponseFormatT')
type SamplingParams = dict[str, int | float | str | bool | list[str] | None]
type Client = OpenAIClient | GoogleClient | AnthropicClient

logger = get_logger(__name__)


class AsyncLLMDispatcher:
    """Core engine: Manages concurrent LLM requests natively using anyio."""

    def __init__(self) -> None:
        """."""
        self._configs: dict[str, EndpointConfig] = {}
        self._clients: dict[str, Client] = {}

    async def __aenter__(self) -> Self:
        """."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        """."""
        await self.close()

    async def close(self) -> None:
        """Gracefully closes all active HTTPX client sessions."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
        logger.info('All AsyncLLMDispatcher connections closed.')

    def register_endpoint(
        self,
        name: str,
        provider: Literal['openai', 'anthropic', 'google'],
        base_url: str | None = None,
        api_key: str | None = None,
        max_concurrent: int = 5,
        max_connections: int = 100,
        max_keepalive: int = 20,
        keepalive_expiry: float | None = None,
        timeout: float = 60.0,
        ssl_verify: bool = True,
    ) -> None:
        """Register an LLM endpoint configuration."""
        self._configs[name] = EndpointConfig(
            name=name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
            ssl_verify=ssl_verify,
        )
        logger.info(f"Registered async endpoint: '{name}' (provider: {provider})")

    def _get_or_create_client(self, endpoint_name: str) -> Client:
        """Lazy-loads the LLM client on first use."""
        if endpoint_name in self._clients:
            return self._clients[endpoint_name]

        config = self._configs[endpoint_name]
        logger.info(f"Instantiating new {config.provider} client for endpoint: '{endpoint_name}'")

        client = create_client(
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            max_concurrent=config.max_concurrent,
            max_connections=config.max_connections,
            max_keepalive=config.max_keepalive,
            keepalive_expiry=config.keepalive_expiry,
            timeout=config.timeout,
            ssl_verify=config.ssl_verify,
        )

        self._clients[endpoint_name] = client

        return client

    @overload
    async def submit(
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
    ) -> ChatCompletionResponse[None]: ...

    @overload
    async def submit(
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
    ) -> ChatCompletionResponse[ResponseFormatT]: ...

    async def submit(
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
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Asynchronously executes an LLM request."""
        if endpoint_name not in self._configs:
            raise ValueError(f"Endpoint '{endpoint_name}' not registered.")

        client = self._get_or_create_client(endpoint_name)

        logger.debug(f"Submitting request to endpoint '{endpoint_name}' using model '{model}'")
        try:
            response = await client.acompletion(
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
        except anyio.get_cancelled_exc_class():
            logger.warning(f"Request to '{endpoint_name}' was cancelled.")
            raise
        # Client specific errors are handled inside the client classes.
        except Exception:
            logger.exception(f"Unexpected error processing LLM request on '{endpoint_name}'")
            raise
        else:
            if response.status == ChatCompletionStatus.SUCCESS:
                logger.debug(
                    f"Request to '{endpoint_name}' successful. "
                    f'Usage: {response.total_tokens if response.total_tokens else "N/A"} tokens'
                )
            elif response.status == ChatCompletionStatus.FAILED:
                logger.debug(f"Request to '{endpoint_name}' failed. Check error logs.")
            return response
