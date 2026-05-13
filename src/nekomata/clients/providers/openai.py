"""OpenAI Client."""

from typing import Any, Literal, TypeVar, overload

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)

from nekomata.clients.base import ClientABC
from nekomata.clients.utils import filter_none
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.types.openai import OpenAIChatCompletionCommonAttrs
from nekomata.utils import get_logger, get_utc_timestamp
from nekomata.utils.uuid import create_uuid

ResponseFormatT = TypeVar('ResponseFormatT')

logger = get_logger(__name__)


class OpenAIClient(ClientABC):
    """OpenAI Client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_concurrent: int | None = None,
        max_connections: int = 100,
        max_keepalive: int = 10,
        keepalive_expiry: float | None = None,
        timeout: float = 60.0,
        ssl_verify: str | bool = True,
    ) -> None:
        """Construct OpenAIClient.

        Args:
            api_key (str | None, optional): OpenAI API key. If None, searches for OPENAI_API_KEY environment.
                Defaults to None.
            base_url (str | None, optional): Base URL for compatible API endpoints. Defaults to None.
            max_concurrent (int | None): Maximum concurrent requests. Defaults to None.
            max_connections (int, optional): Maximum connections per connection pool. Defaults to 100.
            max_keepalive (int, optional): Maximum keep alive connections. Defaults to 10.
            keepalive_expiry (float | None, optional): Keep alive expiration time. Defaults to None.
            timeout (float, optional): Timeout for the client. Defaults to 60.0.
            ssl_verify (str | bool, optional): SSL verification specifications. Defaults to True.

        """
        super(OpenAIClient, self).__init__(
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
            ssl_verify=ssl_verify,
        )

        self._client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client,
        )

        self._initialized = True

    def _extract_common_attrs(self, response: ChatCompletion) -> OpenAIChatCompletionCommonAttrs:
        choices = response.choices
        if len(choices) == 0:
            raise ValueError('Response object has an empty `choices` field.')

        choice = choices[0]
        message = choice.message

        content_string = message.content
        reason_string = (
            message.reasoning_content
            if (
                hasattr(message, 'reasoning_content')
                and (isinstance(message.reasoning_content, str) or message.reasoning_content is None)
            )
            else None
        )

        total_tokens = input_tokens = output_tokens = cache_tokens = reason_tokens = None
        usage = response.usage
        if usage is not None:
            total_tokens = usage.total_tokens
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens

            input_details = usage.prompt_tokens_details
            if input_details is not None:
                cache_tokens = input_details.cached_tokens

            output_details = usage.completion_tokens_details
            if output_details is not None:
                reason_tokens = output_details.reasoning_tokens

        finish_reason = choice.finish_reason

        common_attrs = OpenAIChatCompletionCommonAttrs(
            content_string,
            reason_string,
            finish_reason,
            total_tokens,
            input_tokens,
            output_tokens,
            cache_tokens,
            reason_tokens,
        )
        return common_attrs

    def _convert_create_output(
        self, response: ChatCompletion, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None]:
        common_attrs = self._extract_common_attrs(response=response)
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[None](
            id=id, created_at=created_at, elapsed=elapsed, original=response, **common_attrs._asdict()
        )
        return converted_response

    def _convert_parse_output(
        self, response: ParsedChatCompletion[ResponseFormatT], created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[ResponseFormatT]:
        common_attrs = self._extract_common_attrs(response=response)
        parsed = response.choices[0].message.parsed
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[ResponseFormatT](
            id=id, created_at=created_at, elapsed=elapsed, original=response, parsed=parsed, **common_attrs._asdict()
        )
        return converted_response

    @overload
    def convert_output(
        self, response: ChatCompletion, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None]: ...

    @overload
    def convert_output(
        self, response: ParsedChatCompletion[ResponseFormatT], created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[ResponseFormatT]: ...

    def convert_output(
        self,
        response: ChatCompletion | ParsedChatCompletion[ResponseFormatT],
        created_at: float,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Convert output."""
        if isinstance(response, ParsedChatCompletion):
            return self._convert_parse_output(response=response, created_at=created_at, custom_id=custom_id)
        else:
            return self._convert_create_output(response=response, created_at=created_at, custom_id=custom_id)

    async def _acompletion(
        self,
        created_at: float,
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
        reasoning_effort: Literal['high', 'medium', 'low', 'minimal'] | None = None,
        response_format: type[ResponseFormatT] | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Call OpenAI compatible API."""
        # Construct messages object.
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        # Include unsupported parameters as extra_body.
        if extra_body is None:
            extra_body = {}
        if 'top_k' not in extra_body and top_k is not None:
            extra_body['top_k'] = top_k
        # Filter out empty parameters.
        openai_unsupported_kwargs = filter_none(extra_body) or None

        if response_format is None:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                presence_penalty=presence_penalty,
                max_completion_tokens=max_output_tokens,
                frequency_penalty=frequency_penalty,
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
                extra_body=openai_unsupported_kwargs,
            )
            return self.convert_output(response=response, created_at=created_at, custom_id=custom_id)
        else:
            response = await self._client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                presence_penalty=presence_penalty,
                max_completion_tokens=max_output_tokens,
                frequency_penalty=frequency_penalty,
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                reasoning_effort=reasoning_effort,
                extra_body=openai_unsupported_kwargs,
            )
            return self.convert_output(response=response, created_at=created_at, custom_id=custom_id)
