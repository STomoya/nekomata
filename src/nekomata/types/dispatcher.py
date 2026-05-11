"""Dispatcher."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class EndpointConfig(BaseModel):
    """Endpoint configuration model."""

    name: str
    provider: Literal['openai', 'anthropic', 'google']
    base_url: str | None
    api_key: str | None
    max_concurrent: int
    max_connections: int
    max_keepalive: int
    keepalive_expiry: float | None
    timeout: float
    ssl_verify: bool

    model_config = ConfigDict(
        frozen=True,
    )
