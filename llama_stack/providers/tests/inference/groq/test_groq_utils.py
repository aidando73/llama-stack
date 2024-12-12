from llama_stack.apis.inference import ChatCompletionRequest, UserMessage, SystemMessage, CompletionMessage, StopReason
from llama_stack.providers.remote.inference.groq.groq_utils import convert_chat_completion_request, _convert_message


class TestConvertChatCompletionRequest:
    def test_sets_main_parameters(self):
        request = self._dummy_chat_completion_request()
        request.model = "Llama-3.2-3B"
        request.messages = [UserMessage(content="Hello World")]

        converted = convert_chat_completion_request(request)

        assert converted["model"] == "Llama-3.2-3B"
        assert converted["messages"] == [{"role": "user", "content": "Hello World"},]

    def test_converts_system_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [SystemMessage(content="You are a helpful assistant.")]

        converted = convert_chat_completion_request(request)

        messages = converted["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
    
    def test_converts_completion_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [
            UserMessage(content="Hello World"),
            CompletionMessage(content="Hello World! How can I help you today?", stop_reason=StopReason.end_of_message)
        ]
        

        converted = convert_chat_completion_request(request)

        messages = converted["messages"]
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello World! How can I help you today?"

    def _dummy_chat_completion_request(self):
        return ChatCompletionRequest(
            model="Llama-3.2-3B",
            messages=[UserMessage(content="Hello World")],
        )
