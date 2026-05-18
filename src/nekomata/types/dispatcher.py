"""Dispatcher."""

from pydantic import BaseModel, ConfigDict


class EndpointConfig(BaseModel):
    """Endpoint configuration model."""

    name: str
    provider: str
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
