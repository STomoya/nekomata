<p align="center">
  <img src="asset/nekomata-logo-20260512.png" width="250" alt="Nekomata Logo">
</p>

# Nekomata 🐱

Nekomata is a unified LLM (Large Language Model) dispatcher for Python, designed to provide a consistent interface for multiple providers like OpenAI, Anthropic, and Google (Gemini). It handles concurrent requests, provides structured outputs, and supports both asynchronous and synchronous execution.

## Features

- **Unified API**: Single interface for OpenAI, Anthropic, and Google Gemini.
- **Async & Sync Support**: Native `anyio`-based async dispatcher and a seamless sync adapter.
- **Concurrency Management**: Built-in control for concurrent requests and connection pooling.
- **Structured Outputs**: Easy integration with Pydantic for type-safe structured LLM responses.
- **Consistent Responses**: Unified `ChatCompletionResponse` format across all providers.
- **Lazy Loading**: Clients are instantiated only when first needed.

## Installation

```bash
pip install git+https://github.com/STomoya/nekomata
```

Or with `uv`:

```bash
uv add git+https://github.com/STomoya/nekomata
```

## Quick Start

### Asynchronous Usage

```python
import asyncio
from nekomata import AsyncLLMDispatcher

async def main():
    async with AsyncLLMDispatcher() as dispatcher:
        # Register an endpoint
        dispatcher.register_endpoint(
            name="my-openai",
            provider="openai",
            api_key="your-api-key"
        )

        # Submit a request
        response = await dispatcher.submit(
            endpoint_name="my-openai",
            model="gpt-5.5-pro",
            prompt="Tell me a joke about cats."
        )

        print(f"Response: {response.content}")
        print(f"Usage: {response.total_tokens} tokens")

if __name__ == "__main__":
    asyncio.run(main())
```

### Synchronous Usage

```python
from nekomata import SyncLLMDispatcher

def main():
    with SyncLLMDispatcher() as dispatcher:
        dispatcher.register_endpoint(
            name="my-anthropic",
            provider="anthropic",
            api_key="your-api-key"
        )

        # Submit returns a concurrent.futures.Future
        future = dispatcher.submit(
            endpoint_name="my-anthropic",
            model="claude-sonnet-4.6",
            prompt="Explain why cats always land on their feet in one sentence."
        )

        # Wait for the result
        response = future.result()
        print(f"Response: {response.content}")

if __name__ == "__main__":
    main()
```

## Structured Outputs

Nekomata makes it easy to get structured data back from LLMs using Pydantic models.

```python
from pydantic import BaseModel
from nekomata import SyncLLMDispatcher

class CatInfo(BaseModel):
    breed: str
    origin: str
    temperament: str

with SyncLLMDispatcher() as dispatcher:
    dispatcher.register_endpoint(name="openai", provider="openai")

    future = dispatcher.submit(
        endpoint_name="openai",
        model="gpt-5.5-pro",
        prompt="Tell me about the Maine Coon cat breed.",
        response_format=CatInfo
    )

    response = future.result()
    # Access parsed data
    cat = response.parsed
    print(f"The {cat.breed} originates from {cat.origin}. It is {cat.temperament}.")
```

## Reasoning Support

Nekomata supports models with reasoning capabilities (like OpenAI o1 or Gemini Thinking).

```python
response = await dispatcher.submit(
    endpoint_name="openai",
    model="gpt-5.5-pro",
    prompt="Plan a 24-hour itinerary for a very busy indoor cat.",
    reasoning_effort="high"
)

print(f"Reasoning: {response.reason}")
print(f"Final Answer: {response.content}")
```

## Concurrent Requests & Reordering

When submitting multiple requests, they may complete in a different order than they were submitted. You can use the `custom_id` parameter to track and reorder your responses using `concurrent.futures.as_completed`.

```python
import concurrent.futures
from nekomata import SyncLLMDispatcher

prompts = {
    "meow1": "Write a haiku about a cat sleeping in a sunbeam.",
    "meow2": "Write a haiku about a cat chasing a laser pointer.",
    "meow3": "Write a haiku about a cat demanding treats."
}

with SyncLLMDispatcher() as dispatcher:
    dispatcher.register_endpoint(name="openai", provider="openai")

    # Submit all tasks with a custom_id
    futures = [
        dispatcher.submit(
            endpoint_name="openai",
            model="gpt-5.5-pro",
            prompt=p,
            custom_id=id
        )
        for id, p in prompts.items()
    ]

    # Process as they complete
    results = {}
    for future in concurrent.futures.as_completed(futures):
        response = future.result()
        results[response.id] = response.content

# Outputs can now be accessed in the original order or by ID
for meow_id in prompts.keys():
    print(f"[{meow_id}]: {results[meow_id]}")
```

## Local / Custom Endpoints

Nekomata works seamlessly with OpenAI-compatible servers like **Ollama**, **vLLM**, or **LiteLLM**. Simply register an endpoint with the `openai` provider and your custom `base_url`.

```python
with AsyncLLMDispatcher() as dispatcher:
    dispatcher.register_endpoint(
        name="local-llama",
        provider="openai",
        base_url="http://localhost:11434/v1", # Ollama default
        api_key="ollama" # Usually required but ignored by local servers
    )

    response = await dispatcher.submit(
        endpoint_name="local-llama",
        model="llama3.1",
        prompt="What is a cat's favorite hobby?"
    )
```

## Custom Clients

Implement your own async client class. See the [abstract class](src/nekomata/clients/base.py) and [actual client implementations](src/nekomata/clients/providers/openai.py) for more details.

```python
from typing import Any, Literal, TypeVar

from llm_package_of_your_choice import AsyncClient

from nekomata import register_client
from nekomata.clients.base import ClientABC
from nekomata.types import ChatCompletionStatus
from nekomata.types.clients import PackageSpecificArgs
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.utils import create_uuid, get_utc_timestamp

ResponseFormatT = TypeVar('ResponseFormatT')
PackageArgsT = TypeVar('PackageArgsT', bound=PackageSpecificArgs)

# Register your client to the package using the decorator.
# This will make your client implementation be used inside the dispatcher
# by using the registered name as the provider argument.
@register_client(name="myclient")
class MyClient(ClientABC):

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int | None = None,
        max_connections: int | None = 100,
        max_keepalive: int | None = 10,
        keepalive_expiry: float | None = None,
        timeout: float | None = 60.0,
        ssl_verify: str | bool = True,
    ):
        # Parent __init__ creates a httpx client.
        super(MyClient, self).__init__(
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
            ssl_verify=ssl_verify,
        )

        self._client = AsyncClient(
            ...
            httpx_client=self._http_client,  # Use the httpx client.
        )

        self._initialized = True  # Mark as initialized (currently not used).

    # This is an abstractmethod. Should support the below arguments and return a
    # ChatCompletionResponse object.
    async def _acompletion(
        self,
        created_at: float,
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
        args: PackageArgsT | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        # You only need to implement the inference and conversion to the ChatCompletionResponse object.
        # Structured output retrying, errors, and maximum concurrent requests are already done for you.

        # Do inference here...
        # response = await self._client.chat.completions.create(...)
        content = 'generated content'
        reason = 'thinking...'
        finish_reason = 'stop'
        parsed: ResponseFormatT | None = None  # For structured output.
        input_tokens = 10
        output_tokens = 10
        total_tokens = input_tokens + output_tokens
        cache_tokens = 5
        reason_tokens = 5

        return ChatCompletionResponse(
            id=custom_id or create_uuid(),
            created_at=created_at,
            elapsed=get_utc_timestamp() - created_at,
            status=ChatCompletionStatus.SUCCESS,
            fail_reason=None,
            original=None,  # Place unmodified, original response here.
            content=content,
            reason=reason,
            finish_reason=finish_reason,
            parsed=parsed,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_tokens,
            reason_tokens=reason_tokens,
        )

```

## Configuration

When registering an endpoint, you can configure several parameters:

- `name`: Unique identifier for the endpoint configuration.
- `provider`: The LLM provider (`"openai"`, `"anthropic"`, or `"google"`).
- `api_key`: Provider API key. If not provided, the dispatcher will look for standard environment variables (e.g., `OPENAI_API_KEY`).
- `base_url`: Optional base URL (useful for local proxies like Ollama, LiteLLM, or vLLM).
- `max_concurrent`: Maximum number of concurrent requests allowed for this specific endpoint.
- `timeout`: Request timeout in seconds (default: `60.0`).
- `max_connections`: Maximum number of total HTTP connections in the pool (default: `100`).
- `max_keepalive`: Maximum number of keep-alive connections to maintain (default: `20`).
- `keepalive_expiry`: Time in seconds to keep idle connections alive in the pool.
- `ssl_verify`: Whether to verify SSL certificates (default: `True`).

## Supported Providers

| Provider | Client Package | Models (Examples) |
|----------|----------------|-------------------|
| OpenAI | `openai` | `gpt-5.5-pro`, `gpt-5.5-instant`, `gpt-5.5-cyber` |
| Anthropic | `anthropic` | `claude-opus-4.7`, `claude-sonnet-4.6` |
| Google | `google-genai` | `gemini-3.1-pro`, `gemini-3.1-flash-lite` |

## License

Apache License 2.0
