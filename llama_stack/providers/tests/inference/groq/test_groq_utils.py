import pytest

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    UserMessage,
    SystemMessage,
    CompletionMessage,
    StopReason,
    SamplingStrategy,
)
from llama_stack.providers.remote.inference.groq.groq_utils import (
    convert_chat_completion_request,
    _convert_message,
)


class TestConvertChatCompletionRequest:
    def test_sets_model(self):
        request = self._dummy_chat_completion_request()
        request.model = "Llama-3.2-3B"

        converted = convert_chat_completion_request(request)

        assert converted["model"] == "Llama-3.2-3B"

    def test_converts_user_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [UserMessage(content="Hello World")]

        converted = convert_chat_completion_request(request)

        assert converted["messages"] == [
            {"role": "user", "content": "Hello World"},
        ]

    def test_converts_system_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [SystemMessage(content="You are a helpful assistant.")]

        converted = convert_chat_completion_request(request)

        assert converted["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def test_converts_completion_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [
            UserMessage(content="Hello World"),
            CompletionMessage(
                content="Hello World! How can I help you today?",
                stop_reason=StopReason.end_of_message,
            ),
        ]

        converted = convert_chat_completion_request(request)

        assert converted["messages"] == [
            {"role": "user", "content": "Hello World"},
            {"role": "assistant", "content": "Hello World! How can I help you today?"},
        ]

    def test_does_not_include_logprobs(self):
        request = self._dummy_chat_completion_request()
        request.logprobs = True

        with pytest.warns(Warning) as warnings:
            converted = convert_chat_completion_request(request)

        assert "logprobs are not supported yet" in warnings[0].message.args[0]
        assert converted.get("logprobs") is None

    def test_does_not_include_response_format(self):
        request = self._dummy_chat_completion_request()
        request.response_format = {
            "type": "json_object",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                },
            },
        }

        with pytest.warns(Warning) as warnings:
            converted = convert_chat_completion_request(request)

        assert "response_format is not supported yet" in warnings[0].message.args[0]
        assert converted.get("response_format") is None

    def test_does_not_include_repetition_penalty(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.repetition_penalty = 1.5

        with pytest.warns(Warning) as warnings:
            converted = convert_chat_completion_request(request)

        assert "repetition_penalty is not supported yet" in warnings[0].message.args[0]
        assert converted.get("repetition_penalty") is None
        assert converted.get("frequency_penalty") is None


    def test_includes_stream(self):
        request = self._dummy_chat_completion_request()
        request.stream = True

        converted = convert_chat_completion_request(request)

        assert converted["stream"] is True
    
    def test_n_is_1(self):
        request = self._dummy_chat_completion_request()

        converted = convert_chat_completion_request(request)

        assert converted["n"] == 1
    
    def test_if_max_tokens_is_0_then_it_is_not_included(self):
        request = self._dummy_chat_completion_request()
        # 0 is the default value for max_tokens
        # So we assume that if it's 0, the user didn't set it
        request.sampling_params.max_tokens = 0

        converted = convert_chat_completion_request(request)

        assert converted.get("max_tokens") is None
    
    def test_includes_max_tokens_if_set(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.max_tokens = 100

        converted = convert_chat_completion_request(request)

        assert converted["max_tokens"] == 100

    def _dummy_chat_completion_request(self):
        return ChatCompletionRequest(
            model="Llama-3.2-3B",
            messages=[UserMessage(content="Hello World")],
        )

    def test_includes_temperature(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.temperature = 0.5

        converted = convert_chat_completion_request(request)

        assert converted["temperature"] == 0.5

    def test_includes_top_p(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.top_p = 0.95

        converted = convert_chat_completion_request(request)

        assert converted["top_p"] == 0.95

    