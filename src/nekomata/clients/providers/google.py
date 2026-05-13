"""Google AI Studio Client.

NOTE(stomoya): Currently, we have no plan to support the vertexai version.
"""

from typing import Any, Literal, TypeVar

from google.genai import Client, types
from google.genai.types import GenerateContentResponse
from pydantic import BaseModel

from nekomata.clients.base import ClientABC
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.utils import get_logger, get_utc_timestamp
from nekomata.utils.uuid import create_uuid

ResponseFormatT = TypeVar('ResponseFormatT')

logger = get_logger(__name__)


class GoogleClient(ClientABC):
    """Google Cloud Client."""

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int | None = None,
        max_connections: int = 100,
        max_keepalive: int = 10,
        keepalive_expiry: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Construct GoogleClient.

        Args:
            api_key (str | None, optional): Gemini API key. If None, searches for GEMINI_API_KEY/GOOGLE_API_KEY
                environment variable. Defaults to None.
            max_concurrent (int | None): Maximum concurrent requests. Defaults to None.
            max_connections (int, optional): Maximum connections per connection pool. Defaults to 100.
            max_keepalive (int, optional): Maximum keep alive connections. Defaults to 10.
            keepalive_expiry (float | None, optional): Keep alive expiration time. Defaults to None.
            timeout (float, optional): Timeout for the client. Defaults to 60.0.

        """
        super(GoogleClient, self).__init__(
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
            # NOTE(stomoya): Only used for accessing local unverified SSL endpoints. Hardcoding to True.
            ssl_verify=True,
        )

        self._client: Client = Client(
            api_key=api_key,
            http_options=types.HttpOptions(httpx_async_client=self._http_client),
        )

        self._initialized = True

    def _convert_output(
        self,
        response: GenerateContentResponse,
        created_at: float,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse:
        """Convert output."""
        parsed = response.parsed

        candidates = response.candidates
        if candidates is None or len(candidates) == 0:
            raise ValueError('Response object has an empty `candidates` field.')

        candidate = candidates[0]
        finish_reason = candidate.finish_reason.value if candidate.finish_reason is not None else None

        content = candidate.content
        if content is None or not content.parts:
            raise ValueError('Response object has an candidate with empty content.')

        content_strings: list[str] = []
        reason_strings: list[str] = []
        for part in content.parts:
            # Skip multimodel outputs. (Making type checkers happy.)
            if part.text is None:
                continue

            if part.thought:
                reason_strings.append(part.text)
            else:
                content_strings.append(part.text)
        content_string = ''.join(content_strings) if len(content_strings) > 0 else None
        reason_string = ''.join(reason_strings) if len(reason_strings) > 0 else None

        usage_metadata = response.usage_metadata
        if usage_metadata is not None:
            total_tokens = usage_metadata.total_token_count
            input_tokens = usage_metadata.prompt_token_count
            output_tokens = usage_metadata.candidates_token_count

            cache_tokens = usage_metadata.cached_content_token_count
            reason_tokens = usage_metadata.thoughts_token_count
        else:
            total_tokens = input_tokens = output_tokens = cache_tokens = reason_tokens = None

        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse(
            id=id,
            created_at=created_at,
            elapsed=elapsed,
            original=response,
            content=content_string,
            finish_reason=finish_reason,
            reason=reason_string,
            parsed=parsed,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_tokens,
            reason_tokens=reason_tokens,
        )
        return converted_response

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
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Async completion API call."""
        thinking_config = types.ThinkingConfig(
            include_thoughts=reasoning_effort is not None,
            # NOTE(stomoya): genai package defines a case insensitive enum for this argument.
            #       We simply make the enum raise the error for us.
            thinking_level=reasoning_effort,  # ty: ignore[invalid-argument-type]
        )
        generate_content_config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
            response_schema=response_format,
            response_mime_type='application/json' if response_format else 'text/plain',
            thinking_config=thinking_config,
        )

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=generate_content_config,
        )
        converted_response = self._convert_output(response=response, created_at=created_at, custom_id=custom_id)

        # Raise silently ignored ValidationError by re-validating response JSON schema.
        if (
            # structured output is used
            response_format is not None
            # but doesn't have a parsed object
            and converted_response.parsed is None
            # and seems to have succeeded generating the content.
            and converted_response.content is not None
            and issubclass(response_format, BaseModel)  # happy type chcker
        ):
            response_format.model_validate_json(converted_response.content)

        return converted_response
