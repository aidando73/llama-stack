from llama_stack.apis.inference import ChatCompletionRequest, Message, Role

from groq.types.chat.completion_create_params import CompletionCreateParams
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from groq.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from groq.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from groq.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)


def convert_chat_completion_request(
    request: ChatCompletionRequest,
) -> CompletionCreateParams:
    """
    Convert a ChatCompletionRequest to a Groq API-compatible dictionary.
    """

    return CompletionCreateParams(
        model=request.model,
        messages=[_convert_message(message) for message in request.messages],
        sampling_params="testing",
    )


def _convert_message(message: Message) -> ChatCompletionMessageParam:
    if message.role == Role.system.value:
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    elif message.role == Role.user.value:
        return ChatCompletionUserMessageParam(role="user", content=message.content)
    elif message.role == Role.assistant.value:
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=message.content
        )
    else:
        raise ValueError(f"Invalid message role: {message.role}")
