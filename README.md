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
