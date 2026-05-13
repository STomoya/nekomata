"""Anthropic Client."""

from typing import Any, Literal, TypeVar, overload

from anthropic import AsyncAnthropic, Omit
from anthropic.types import (
    Message,
    MessageParam,
    OutputConfigParam,
    ParsedMessage,
    TextBlock,
    ThinkingBlock,
    ThinkingConfigAdaptiveParam,
    ThinkingConfigParam,
)

from nekomata.clients.base import ClientABC
from nekomata.types.anthropic import AnthropicMessagesCommonAttrs
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.utils import get_logger, get_utc_timestamp
from nekomata.utils.uuid import create_uuid

ResponseFormatT = TypeVar('ResponseFormatT')

logger = get_logger(__name__)


class AnthropicClient(ClientABC):
    """Anthropic Client."""

    def __init__(
        self,
        api_key: str | None = None,
        max_concurrent: int | None = None,
        max_connections: int = 100,
        max_keepalive: int = 10,
        keepalive_expiry: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Construct AnthropicClient.

        Args:
            api_key (str | None, optional): Anthropic API key. If None, searches for ANTHROPIC_API_KEY environment.
                Defaults to None.
            max_concurrent (int | None): Maximum concurrent requests. Defaults to None.
            max_connections (int, optional): Maximum connections per connection pool. Defaults to 100.
            max_keepalive (int, optional): Maximum keep alive connections. Defaults to 10.
            keepalive_expiry (float | None, optional): Keep alive expiration time. Defaults to None.
            timeout (float, optional): Timeout for the client. Defaults to 60.0.

        """
        super(AnthropicClient, self).__init__(
            max_concurrent=max_concurrent,
            max_connections=max_connections,
            max_keepalive=max_keepalive,
            keepalive_expiry=keepalive_expiry,
            timeout=timeout,
            # NOTE(stomoya): Only used for accessing local unverified SSL endpoints. Hardcoding to True.
            ssl_verify=True,
        )

        self._client: AsyncAnthropic = AsyncAnthropic(
            api_key=api_key,
            http_client=self._http_client,
        )

        self._initialized = True

    def _extract_common_attrs(self, response: Message) -> AnthropicMessagesCommonAttrs:
        finish_reason = response.stop_reason

        contents = response.content
        if len(contents) == 0:
            raise ValueError('Response object has an empty `content` field.')

        content_strings: list[str] = []
        reason_strings: list[str] = []
        for content in contents:
            if isinstance(content, TextBlock):
                content_strings.append(content.text)
            elif isinstance(content, ThinkingBlock):
                reason_strings.append(content.thinking)
        content_string = ''.join(content_strings) if len(content_strings) > 0 else None
        reason_string = ''.join(reason_strings) if len(reason_strings) > 0 else None

        usage = response.usage
        # It seems we don't have a way to collect reason tokens information.
        reason_tokens = None
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        total_tokens = input_tokens + output_tokens
        cache_tokens = usage.cache_read_input_tokens

        common_attrs = AnthropicMessagesCommonAttrs(
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

    def _convert_create_response(
        self, response: Message, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None]:
        common_attrs = self._extract_common_attrs(response)
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[None](
            id=id, created_at=created_at, elapsed=elapsed, original=response, **common_attrs._asdict()
        )
        return converted_response

    def _convert_parse_response(
        self, response: ParsedMessage[ResponseFormatT], created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[ResponseFormatT]:
        common_attrs = self._extract_common_attrs(response)
        parsed = response.parsed_output
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[ResponseFormatT](
            id=id, created_at=created_at, elapsed=elapsed, original=response, parsed=parsed, **common_attrs._asdict()
        )
        return converted_response

    @overload
    def convert_output(
        self, response: ParsedMessage, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[ResponseFormatT]: ...

    @overload
    def convert_output(
        self, response: Message, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None]: ...

    def convert_output(
        self, response: Message | ParsedMessage[ResponseFormatT], created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Convert output."""
        if isinstance(response, ParsedMessage):
            return self._convert_parse_response(response=response, created_at=created_at, custom_id=custom_id)
        else:
            return self._convert_create_response(response=response, created_at=created_at, custom_id=custom_id)

    @overload
    async def acompletion(
        self,
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
    async def acompletion(
        self,
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

    async def acompletion(
        self,
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
        """Call anthropic messages API.

        `max_output_tokens` is required by the messages API. Will default to 4096 if not provided.
        `reasoning_effort='minimal'` is not supported. Will fallback to disabled.
        `temperature`, `top_p`, `top_k` are deperecated by Anthropic and will be ignored.
        `presence_penalty`, `frequency_penalty`, `seed`, `extra_body` are unsupported by Anthropic and will be ignored.

        Args:
            model (str): The name of the model. Example: `gpt-5`.
            prompt (str): User input.
            system_prompt (str | None, optional): System prompt. Defaults to None.
            max_output_tokens (str | None, optional): Maximum output tokens. Defaults to None.
            temperature (float | None, optional): IGNORED. Defaults to None.
            top_p (float | None, optional): IGNORED. Defaults to None.
            top_k (int | None, optional): IGNORED. Defaults to None.
            presence_penalty (float | None, optional): IGNORED. Defaults to None.
            frequency_penalty (float | None, optional): IGNORED. Defatuls to None.
            seed (int | None): IGNORED. Defaults to None.
            response_format (type[BaseModel] | None, optional): JSON response format defined as a pydantic model.
                Defaults to None.
            reasoning_effort (Literal['high', 'medium', 'low', 'minimal'] | None, optional): Reasoning effort.
                Defaults to None.
            extra_body (dict[str, Any] | None, optional): IGNORED.
            custom_id (str | None, optional): Custom ID. This value will overwrite the response object's ID field.

        """
        # Construct messages object.
        messages: list[MessageParam] = [{'role': 'user', 'content': prompt}]

        if max_output_tokens is None:
            # TODO: show warning.
            max_output_tokens = 4096

        thinking = output_config = None
        if reasoning_effort == 'minimal':
            # TODO: warn unsupported effort 'minimal' will fallback to disabled.
            pass
        elif reasoning_effort:
            # Maybe support the deprecated ThinkingConfigEnabledParam for older Claude models?
            thinking: ThinkingConfigParam = ThinkingConfigAdaptiveParam(type='adaptive', display='summarized')
            output_config: OutputConfigParam = OutputConfigParam(effort=reasoning_effort)

        # TODO: Log deprecated/unsupport parameters if set.
        #   deprecated: temperature, top_p, top_k
        #   unsupported: presence_penalty, frequency_penalty, seed, extra_body

        omit = Omit()

        logger.debug(f'Entering semaphore for model: {model}')
        async with self.semaphore:
            logger.debug(f'Acquired semaphore for model: {model}')
            created_at = get_utc_timestamp()

            try:
                if response_format is None:
                    response = await self._client.messages.create(
                        max_tokens=max_output_tokens,
                        messages=messages,
                        model=model,
                        stream=False,
                        system=system_prompt or omit,
                        thinking=thinking or omit,
                        output_config=output_config or omit,
                    )
                    return self.convert_output(response=response, created_at=created_at, custom_id=custom_id)
                else:
                    response = await self._client.messages.parse(
                        max_tokens=max_output_tokens,
                        messages=messages,
                        model=model,
                        stream=False,
                        system=system_prompt or omit,
                        thinking=thinking or omit,
                        output_config=output_config or omit,
                        output_format=response_format,
                    )
                    return self.convert_output(response=response, created_at=created_at, custom_id=custom_id)
            except Exception as e:
                return self.handle_exception(
                    err_msg='Anthropic API call failed', exc=e, created_at=created_at, custom_id=custom_id
                )
