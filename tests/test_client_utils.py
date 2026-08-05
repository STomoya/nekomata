"""Tests for the client utilities and polling functions."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from nekomata.clients.plugins.anthropic import AnthropicBatchAPIPlugin
from nekomata.clients.plugins.google import GoogleBatchAPIPlugin
from nekomata.clients.plugins.openai import OpenAIBatchAPIPlugin
from nekomata.clients.helpers import apoll_batch
from nekomata.clients.utils import filter_none, validate_and_expand_batch_args


class MockOpenAIClient(OpenAIBatchAPIPlugin):
    """Mock client for OpenAI batch plugin tests."""

    def __init__(self, statuses: list[str]) -> None:
        """Construct mock client."""
        self.statuses = list(statuses)
        self.calls = 0

    async def acreate_batch(self, **kwargs: Any) -> Any:
        """Create mock batch."""
        return 'mock'

    async def aretrieve_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Retrieve mock batch."""
        self.calls += 1
        res = MagicMock()
        res.status = self.statuses.pop(0)
        return res

    async def acancel_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Cancel mock batch."""
        return 'mock'

    async def alist_batches(self, *args: Any, **kwargs: Any) -> Any:
        """List mock batches."""
        return 'mock'

    async def adelete_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Delete mock batch."""
        return 'mock'


class MockAnthropicClient(AnthropicBatchAPIPlugin):
    """Mock client for Anthropic batch plugin tests."""

    def __init__(self, statuses: list[str]) -> None:
        """Construct mock client."""
        self.statuses = list(statuses)
        self.calls = 0

    async def acreate_batch(self, **kwargs: Any) -> Any:
        """Create mock batch."""
        return 'mock'

    async def aretrieve_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Retrieve mock batch."""
        self.calls += 1
        res = MagicMock()
        res.processing_status = self.statuses.pop(0)
        return res

    async def acancel_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Cancel mock batch."""
        return 'mock'

    async def alist_batches(self, *args: Any, **kwargs: Any) -> Any:
        """List mock batches."""
        return 'mock'

    async def adelete_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Delete mock batch."""
        return 'mock'


class MockGoogleClient(GoogleBatchAPIPlugin):
    """Mock client for Google batch plugin tests."""

    def __init__(self, statuses: list[Any]) -> None:
        """Construct mock client."""
        self.statuses = list(statuses)
        self.calls = 0

    async def acreate_batch(self, **kwargs: Any) -> Any:
        """Create mock batch."""
        return 'mock'

    async def aretrieve_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Retrieve mock batch."""
        self.calls += 1
        res = MagicMock()
        res.state = self.statuses.pop(0)
        return res

    async def acancel_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Cancel mock batch."""
        return 'mock'

    async def alist_batches(self, *args: Any, **kwargs: Any) -> Any:
        """List mock batches."""
        return 'mock'

    async def adelete_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Delete mock batch."""
        return 'mock'


class MockCustomClient:
    """Mock custom client that does not inherit from any batch plugin class."""

    def __init__(self, statuses: list[Any]) -> None:
        """Construct mock client."""
        self.statuses = list(statuses)
        self.calls = 0

    async def aretrieve_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
        """Retrieve mock batch."""
        self.calls += 1
        return self.statuses.pop(0)


class MockEnum:
    """Mock enum class for testing name/value conversion."""

    def __init__(self, name: str, value: str) -> None:
        """Construct mock enum."""
        self.name = name
        self.value = value


class TestClientUtils:
    """Test suite for client utilities."""

    def test_filter_none(self) -> None:
        """Test filtering none values from dict."""
        d = {'a': 1, 'b': None, 'c': 'hello'}
        assert filter_none(d) == {'a': 1, 'c': 'hello'}

    @pytest.mark.anyio
    async def test_apoll_batch_openai(self) -> None:
        """Test apoll_batch for OpenAI client."""
        client = MockOpenAIClient(['validating', 'in_progress', 'completed'])
        res = await apoll_batch(client, 'batch-id', interval=0.01)
        assert client.calls == 3
        assert res.status == 'completed'

    @pytest.mark.anyio
    async def test_apoll_batch_anthropic(self) -> None:
        """Test apoll_batch for Anthropic client."""
        client = MockAnthropicClient(['in_progress', 'ended'])
        res = await apoll_batch(client, 'batch-id', interval=0.01)
        assert client.calls == 2
        assert res.processing_status == 'ended'

    @pytest.mark.anyio
    async def test_apoll_batch_google(self) -> None:
        """Test apoll_batch for Google client."""
        client = MockGoogleClient(['JOB_STATE_RUNNING', 'JOB_STATE_SUCCEEDED'])
        res = await apoll_batch(client, 'batch-id', interval=0.01)
        assert client.calls == 2
        assert res.state == 'JOB_STATE_SUCCEEDED'

    @pytest.mark.anyio
    async def test_apoll_batch_google_enum(self) -> None:
        """Test apoll_batch for Google client with Enum states."""
        client = MockGoogleClient(
            [
                MockEnum('JOB_STATE_RUNNING', 'job_state_running'),
                MockEnum('JOB_STATE_SUCCEEDED', 'job_state_succeeded'),
            ]
        )
        res = await apoll_batch(client, 'batch-id', interval=0.01)
        assert client.calls == 2
        assert res.state.name == 'JOB_STATE_SUCCEEDED'

    @pytest.mark.anyio
    async def test_apoll_batch_custom_client_success(self) -> None:
        """Test apoll_batch for custom client with stop_statuses."""
        client = MockCustomClient(['status_a', 'status_b', 'status_c'])
        res = await apoll_batch(client, 'batch-id', stop_statuses=['status_c'], interval=0.01)
        assert client.calls == 3
        assert res == 'status_c'

    @pytest.mark.anyio
    async def test_apoll_batch_custom_client_missing_stop_statuses(self) -> None:
        """Test apoll_batch raises ValueError when stop_statuses is missing for custom clients."""
        client = MockCustomClient(['status_a'])
        with pytest.raises(ValueError, match='stop_statuses must be provided'):
            await apoll_batch(client, 'batch-id', interval=0.01)

    @pytest.mark.anyio
    async def test_apoll_batch_dict_status(self) -> None:
        """Test apoll_batch extracting status from dict result."""
        client = MockCustomClient(
            [
                {'status': 'running'},
                {'status': 'finished'},
            ]
        )
        res = await apoll_batch(client, 'batch-id', stop_statuses=['finished'], interval=0.01)
        assert client.calls == 2
        assert res == {'status': 'finished'}

    @pytest.mark.anyio
    async def test_apoll_batch_invalid_status(self) -> None:
        """Test apoll_batch raises ValueError when status cannot be determined."""
        client = MockCustomClient([object()])
        with pytest.raises(ValueError, match='Could not extract status'):
            await apoll_batch(client, 'batch-id', stop_statuses=['finished'], interval=0.01)

    @pytest.mark.anyio
    async def test_apoll_batch_custom_client_various_attributes(self) -> None:
        """Test apoll_batch for custom clients with status, state, or processing_status attributes."""

        # 1. Custom status attribute
        class CustomObjStatus:
            def __init__(self, status: str) -> None:
                self.status = status

        client1 = MockCustomClient([CustomObjStatus('finished')])
        res1 = await apoll_batch(client1, 'batch-id', stop_statuses=['finished'], interval=0.01)
        assert res1.status == 'finished'

        # 2. Custom state attribute
        class CustomObjState:
            def __init__(self, state: str) -> None:
                self.state = state

        client2 = MockCustomClient([CustomObjState('finished')])
        res2 = await apoll_batch(client2, 'batch-id', stop_statuses=['finished'], interval=0.01)
        assert res2.state == 'finished'

        # 3. Custom processing_status attribute
        class CustomObjProcessingStatus:
            def __init__(self, processing_status: str) -> None:
                self.processing_status = processing_status

        client3 = MockCustomClient([CustomObjProcessingStatus('finished')])
        res3 = await apoll_batch(client3, 'batch-id', stop_statuses=['finished'], interval=0.01)
        assert res3.processing_status == 'finished'

    @pytest.mark.anyio
    async def test_apoll_batch_value_only_enum(self) -> None:
        """Test apoll_batch for enums/objects with value attribute but no name attribute."""

        class CustomObjStatus:
            def __init__(self, status: Any) -> None:
                self.status = status

        class ValEnumOnly:
            def __init__(self, value: str) -> None:
                self.value = value

        client = MockCustomClient([CustomObjStatus(ValEnumOnly('finished'))])
        res = await apoll_batch(client, 'batch-id', stop_statuses=['finished'], interval=0.01)
        assert res.status.value == 'finished'

    @pytest.mark.anyio
    async def test_apoll_batch_google_with_done_property(self) -> None:
        """Test apoll_batch for Google client using its built-in done property."""

        class MockGoogleClientWithDone(GoogleBatchAPIPlugin):
            def __init__(self) -> None:
                self.calls = 0

            async def acreate_batch(self, **kwargs: Any) -> Any:
                return 'mock'

            async def aretrieve_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                res = MagicMock()
                # 1st call: not done, 2nd call: done
                res.done = self.calls == 2
                res.state = 'JOB_STATE_RUNNING' if self.calls == 1 else 'JOB_STATE_SUCCEEDED'
                return res

            async def acancel_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
                return 'mock'

            async def alist_batches(self, *args: Any, **kwargs: Any) -> Any:
                return 'mock'

            async def adelete_batch(self, batch_id: str, *args: Any, **kwargs: Any) -> Any:
                return 'mock'

        client = MockGoogleClientWithDone()
        res = await apoll_batch(client, 'batch-id', interval=0.01)
        assert client.calls == 2
        assert res.done is True


class TestValidateAndExpandBatchArgs:
    """Test suite for validate_and_expand_batch_args utility."""

    def test_all_lists(self) -> None:
        """Test validate_and_expand_batch_args when all arguments are lists."""
        res = validate_and_expand_batch_args(
            prompt=['p1', 'p2'],
            system_prompt=['sys1', 'sys2'],
            max_output_tokens=[10, 20],
            response_format=['format1', 'format2'],
            reasoning_effort=['effort1', 'effort2'],
            custom_id=['id1', 'id2'],
        )
        assert len(res) == 2
        assert res[0].prompt == 'p1'
        assert res[1].prompt == 'p2'
        assert res[0].system_prompt == 'sys1'
        assert res[0].max_output_tokens == 10
        assert res[0].response_format == 'format1'
        assert res[0].reasoning_effort == 'effort1'
        assert res[0].custom_id == 'id1'

    def test_no_lists_raises_error(self) -> None:
        """Test validate_and_expand_batch_args raises error when no lists are provided."""
        with pytest.raises(ValueError, match='must be a list'):
            validate_and_expand_batch_args(
                prompt='prompt',
                system_prompt='sys',
                max_output_tokens=10,
                response_format='format',
                reasoning_effort='effort',
                custom_id='id',
            )

    @pytest.mark.parametrize(
        'prompt,system_prompt,max_output_tokens,response_format,reasoning_effort,custom_id,expected_length',
        [
            # prompt is list, others are non-list
            (['p1', 'p2'], 's1', 10, 'f1', 'r1', 'id', 2),
            # system_prompt is list, others are non-list
            ('p1', ['s1', 's2'], 10, 'f1', 'r1', 'id', 2),
            # max_output_tokens is list, others are non-list
            ('p1', 's1', [10, 20], 'f1', 'r1', 'id', 2),
            # response_format is list, others are non-list
            ('p1', 's1', 10, ['f1', 'f2'], 'r1', 'id', 2),
            # reasoning_effort is list, others are non-list
            ('p1', 's1', 10, 'f1', ['r1', 'r2'], 'id', 2),
            # custom_id is list, others are non-list
            ('p1', 's1', 10, 'f1', 'r1', ['id1', 'id2'], 2),
            # None values tests: list prompt, others are None (where allowed)
            (['p1', 'p2'], None, None, None, None, None, 2),
        ],
    )
    def test_single_list_argument_expansion(
        self,
        prompt: Any,
        system_prompt: Any,
        max_output_tokens: Any,
        response_format: Any,
        reasoning_effort: Any,
        custom_id: Any,
        expected_length: int,
    ) -> None:
        """Test with one list argument and others as single or None values."""
        res = validate_and_expand_batch_args(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            custom_id=custom_id,
        )
        assert len(res) == expected_length

        for i in range(expected_length):
            # Check prompt
            if isinstance(prompt, list):
                assert res[i].prompt == prompt[i]
            else:
                assert res[i].prompt == prompt

            # Check system_prompt
            if isinstance(system_prompt, list):
                assert res[i].system_prompt == system_prompt[i]
            else:
                assert res[i].system_prompt == system_prompt

            # Check max_output_tokens
            if isinstance(max_output_tokens, list):
                assert res[i].max_output_tokens == max_output_tokens[i]
            else:
                assert res[i].max_output_tokens == max_output_tokens

            # Check response_format
            if isinstance(response_format, list):
                assert res[i].response_format == response_format[i]
            else:
                assert res[i].response_format == response_format

            # Check reasoning_effort
            if isinstance(reasoning_effort, list):
                assert res[i].reasoning_effort == reasoning_effort[i]
            else:
                assert res[i].reasoning_effort == reasoning_effort

            # Check custom_id
            if isinstance(custom_id, list):
                assert res[i].custom_id == custom_id[i]
            elif custom_id is not None:
                assert res[i].custom_id == f'{custom_id}-{i}'
            else:
                assert res[i].custom_id.startswith('req-')
