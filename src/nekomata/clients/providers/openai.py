"""OpenAI Client."""

from typing import Any, TypeVar, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ParsedChatCompletion,
)
from openai.types.responses import ParsedResponse, Response, ResponseReasoningItem

from nekomata.clients.base import ClientABC
from nekomata.clients.plugins.openai import OpenAIBatchAPIPlugin
from nekomata.clients.utils import filter_none
from nekomata.types.integrations import ChatCompletionResponse
from nekomata.types.openai import OpenAIArgs, OpenAIChatCompletionCommonAttrs, ResponsesArgs
from nekomata.utils import get_logger, get_utc_timestamp
from nekomata.utils.uuid import create_uuid

ResponseFormatT = TypeVar('ResponseFormatT')

logger = get_logger(__name__)


class OpenAIClient(ClientABC, OpenAIBatchAPIPlugin):
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

    def _extract_chat_completion_common_attrs(self, response: ChatCompletion) -> OpenAIChatCompletionCommonAttrs:
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

    def _convert_chat_completion_create_output(
        self, response: ChatCompletion, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None]:
        common_attrs = self._extract_chat_completion_common_attrs(response=response)
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[None](
            id=id, created_at=created_at, elapsed=elapsed, original=response, **common_attrs._asdict()
        )
        return converted_response

    def _convert_chat_completion_parse_output(
        self, response: ParsedChatCompletion[ResponseFormatT], created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[ResponseFormatT]:
        common_attrs = self._extract_chat_completion_common_attrs(response=response)
        parsed = response.choices[0].message.parsed
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[ResponseFormatT](
            id=id, created_at=created_at, elapsed=elapsed, original=response, parsed=parsed, **common_attrs._asdict()
        )
        return converted_response

    def _convert_chat_completion_output(
        self,
        response: ChatCompletion | ParsedChatCompletion[ResponseFormatT],
        created_at: float,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Convert output."""
        if isinstance(response, ParsedChatCompletion):
            return self._convert_chat_completion_parse_output(
                response=response, created_at=created_at, custom_id=custom_id
            )
        else:
            return self._convert_chat_completion_create_output(
                response=response, created_at=created_at, custom_id=custom_id
            )

    async def _chat_completion(
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
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
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
            response = await self._client.chat.completions.create(  # ty: ignore[ no-matching-overload]
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
            return self._convert_chat_completion_output(response=response, created_at=created_at, custom_id=custom_id)
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
                # NOTE(stomoya): Let the package or API raise the unsupported reasoning_effort value
                reasoning_effort=reasoning_effort,  # ty: ignore[invalid-argument-type]
                extra_body=openai_unsupported_kwargs,
            )
            return self._convert_chat_completion_output(response=response, created_at=created_at, custom_id=custom_id)

    def _extract_responses_common_attrs(self, response: Response) -> OpenAIChatCompletionCommonAttrs:
        """Extract common attrs from the Response object."""
        # NOTE(stomoya): I see no finish reason.
        finish_reason = response.status
        content_string = response.output_text

        # Extract reason string from the contents.
        reason_string = ''
        for output in response.output:
            if output.type == 'reasoning':
                output = cast(ResponseReasoningItem, output)
                reason_string_summary = ''.join(item.text for item in output.summary)
                if output.content is not None:
                    reason_string_content = ''.join(item.text for item in output.content)
                else:
                    reason_string_content = None
                reason_string += reason_string_content or reason_string_summary
        if not reason_string.strip():
            reason_string = None

        # Usage
        usage = response.usage
        total_tokens = input_tokens = output_tokens = cache_tokens = reason_tokens = None
        if usage is not None:
            total_tokens = usage.total_tokens
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            if usage.input_tokens_details is not None:
                cache_tokens = usage.input_tokens_details.cached_tokens
            if usage.output_tokens_details is not None:
                reason_tokens = usage.output_tokens_details.reasoning_tokens

        common_attrs = OpenAIChatCompletionCommonAttrs(
            content=content_string,
            reason=reason_string,
            finish_reason=finish_reason,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_tokens=cache_tokens,
            reason_tokens=reason_tokens,
        )
        return common_attrs

    def _convert_responses_create_output(
        self, response: Response, created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[None]:
        common_attrs = self._extract_responses_common_attrs(response=response)
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[None](
            id=id, created_at=created_at, elapsed=elapsed, original=response, **common_attrs._asdict()
        )
        return converted_response

    def _convert_responses_parse_output(
        self, response: ParsedResponse[ResponseFormatT], created_at: float, custom_id: str | None = None
    ) -> ChatCompletionResponse[ResponseFormatT]:
        common_attrs = self._extract_responses_common_attrs(response=response)
        parsed = response.output_parsed
        id = custom_id or create_uuid()
        elapsed = get_utc_timestamp() - created_at
        converted_response = ChatCompletionResponse[ResponseFormatT](
            id=id, created_at=created_at, elapsed=elapsed, original=response, parsed=parsed, **common_attrs._asdict()
        )
        return converted_response

    def _convert_responses_output(
        self,
        response: Response | ParsedResponse[ResponseFormatT],
        created_at: float,
        custom_id: str | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        if isinstance(response, ParsedResponse):
            return self._convert_responses_parse_output(response=response, created_at=created_at, custom_id=custom_id)
        else:
            return self._convert_responses_create_output(response=response, created_at=created_at, custom_id=custom_id)

    async def _responses(
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
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
        args: ResponsesArgs | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """OpenAI responses API call."""
        args = args or ResponsesArgs()
        if response_format is not None:
            response = await self._client.responses.parse(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                instructions=system_prompt,
                top_p=top_p,
                text_format=response_format,
                reasoning={'effort': reasoning_effort} if reasoning_effort else None,  # ty: ignore[invalid-argument-type]
                store=args.store,
                previous_response_id=args.response_id,
            )
            return self._convert_responses_output(response=response, created_at=created_at, custom_id=custom_id)
        else:
            response = await self._client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                instructions=system_prompt,
                top_p=top_p,
                reasoning={'effort': reasoning_effort} if reasoning_effort else None,  # ty: ignore[invalid-argument-type]
                store=args.store,
                previous_response_id=args.response_id,
            )
            return self._convert_responses_output(response=response, created_at=created_at, custom_id=custom_id)

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
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        custom_id: str | None = None,
        args: OpenAIArgs | None = None,
    ) -> ChatCompletionResponse[None] | ChatCompletionResponse[ResponseFormatT]:
        """Call OpenAI compatible API."""
        if args is None or args.api == 'chat_completions':
            response = await self._chat_completion(
                created_at=created_at,
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
            return response
        elif args.api == 'responses':
            response = await self._responses(
                created_at=created_at,
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
            return response
        else:  # pragma: no cover
            raise ValueError(f'Unknown API variant "{args.api}".')
