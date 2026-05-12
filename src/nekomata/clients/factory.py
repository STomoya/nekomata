"""LLM client factory."""

from typing import Literal, overload

from nekomata.clients.providers.anthropic import AnthropicClient
from nekomata.clients.providers.google import GoogleClient
from nekomata.clients.providers.openai import OpenAIClient
from nekomata.utils import get_logger

Client = OpenAIClient | GoogleClient | AnthropicClient
SUPPORTED_PROVIDERS = {'openai', 'google', 'anthropic'}

logger = get_logger(__name__)


@overload
def create_client(
    provider: Literal['openai'] = 'openai',
    api_key: str | None = None,
    base_url: str | None = None,
    max_concurrent: int | None = None,
    max_connections: int = 100,
    max_keepalive: int = 10,
    keepalive_expiry: float | None = None,
    timeout: float = 60.0,
    ssl_verify: str | bool = True,
) -> OpenAIClient: ...


@overload
def create_client(
    provider: Literal['google'] = 'google',
    api_key: str | None = None,
    base_url: str | None = None,
    max_concurrent: int | None = None,
    max_connections: int = 100,
    max_keepalive: int = 10,
    keepalive_expiry: float | None = None,
    timeout: float = 60.0,
    ssl_verify: str | bool = True,
) -> GoogleClient: ...


@overload
def create_client(
    provider: Literal['anthropic'] = 'anthropic',
    api_key: str | None = None,
    base_url: str | None = None,
    max_concurrent: int | None = None,
    max_connections: int = 100,
    max_keepalive: int = 10,
    keepalive_expiry: float | None = None,
    timeout: float = 60.0,
    ssl_verify: str | bool = True,
) -> AnthropicClient: ...


def create_client(
    provider: Literal['openai', 'google', 'anthropic'],
    api_key: str | None = None,
    base_url: str | None = None,
    max_concurrent: int | None = None,
    max_connections: int = 100,
    max_keepalive: int = 10,
    keepalive_expiry: float | None = None,
    timeout: float = 60.0,
    ssl_verify: str | bool = True,
) -> Client:
    """Create client by provider name."""
    if provider not in SUPPORTED_PROVIDERS:
        err_msg = f'Unsupport provider "{provider}". Must be one of {SUPPORTED_PROVIDERS}'
        raise ValueError(err_msg)

    logger.debug(f"Creating LLM client for provider: '{provider}' (base_url: {base_url})")
    if provider == 'openai':
        return OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
            ssl_verify=ssl_verify,
        )
    elif provider == 'google':
        return GoogleClient(
            api_key=api_key,
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
        )
    elif provider == 'anthropic':
        return AnthropicClient(
            api_key=api_key,
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
        )
    else:  # pragma: no cover # Never reached. Making type checker happy.
        err_msg = f'Unknown provider "{provider}". Must be one of {SUPPORTED_PROVIDERS}'
        raise ValueError(err_msg)
