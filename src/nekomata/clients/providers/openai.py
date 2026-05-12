"""OpenAI Client."""

from typing import Any, Literal, TypeVar, overload

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)

from nekomata.clients.base import ClientABC
from nekomata.clients.utils import create_failed_response, filter_none
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.types.openai import OpenAIChatCompletionCommonAttrs
from nekomata.utils import get_logger, get_utc_timestamp

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

    def _convert_create_output(self, response: ChatCompletion, created_at: float) -> ChatCompletionResponse[None]:
        common_attrs = self._extract_common_attrs(response=response)
        converted_response = ChatCompletionResponse[None](
            created_at=created_at,
            original=response,
            **common_attrs._asdict(),
        )
        return converted_response

    def _convert_parse_output(
        self, response: ParsedChatCompletion[ResponseFormatT], created_at: float
    ) -> ChatCompletionResponse[ResponseFormatT]:
        common_attrs = self._extract_common_attrs(response=response)

        parsed = response.choices[0].message.parsed

        converted_response = ChatCompletionResponse[ResponseFormatT](
            created_at=created_at,
            original=response,
            parsed=parsed,
            **common_attrs._asdict(),
        )
        return converted_response

    @overload
    def convert_output(self, response: ChatCompletion, created_at: float) -> ChatCompletionResponse[None]: ...

    @overload
    def convert_output(
        self, response: ParsedChatCompletion[ResponseFormatT], created_at: float
    ) -> ChatCompletionResponse[ResponseFormatT]: ...

    def convert_output(
        self, response: ChatCompletion | ParsedChatCompletion[ResponseFormatT], created_at: float
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Convert output."""
        if isinstance(response, ParsedChatCompletion):
            return self._convert_parse_output(response, created_at)
        else:
            return self._convert_create_output(response, created_at)

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
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Call OpenAI compatible API.

        Args:
            model (str): The name of the model. Example: `gpt-5`.
            prompt (str): User input.
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

        """
        # Construct messages object.
        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt},
            ]
        messages.append({'role': 'user', 'content': prompt})

        # Include unsupported parameters as extra_body.
        if extra_body is None:
            extra_body = {}
        if 'top_k' not in extra_body and top_k is not None:
            extra_body['top_k'] = top_k
        # Filter out empty parameters.
        openai_unsupported_kwargs = filter_none(extra_body) or None

        logger.debug(f'Entering semaphore for model: {model}')
        async with self.semaphore:
            logger.debug(f'Acquired semaphore for model: {model}')
            created_at = get_utc_timestamp()

            try:
                # NOTE(stomoya): We explicitly pass all the arguments to these functions for type checkers to correctly
                #   resolve the overloads for `.create()`.
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
                    return self.convert_output(response=response, created_at=created_at)
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
                    return self.convert_output(response=response, created_at=created_at)
            except Exception as e:
                logger.exception('OpenAI API call failed')
                return create_failed_response(response=None, fail_reason=f'{e!s}', created_at=created_at)
