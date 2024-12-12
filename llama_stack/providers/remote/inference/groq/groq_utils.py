import warnings

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
    Warns client if request contains unsupported features.
    """

    if request.logprobs:
        # Groq doesn't support logprobs at the time of writing
        warnings.warn("logprobs are not supported yet")
    
    if request.response_format:
        # Groq's JSON mode is beta at the time of writing
        warnings.warn("response_format is not supported yet")
    
    if request.sampling_params.repetition_penalty:
        # groq supports frequency_penalty, but frequency_penalty and sampling_params.repetition_penalty
        # seem to have different semantics
        # frequency_penalty defaults to 0 is a float between -2.0 and 2.0
        # repetition_penalty defaults to 1 and is often set somewhere between 1.0 and 2.0
        # so we exclude it for now
        warnings.warn("repetition_penalty is not supported yet")

    return CompletionCreateParams(
        model=request.model,
        messages=[_convert_message(message) for message in request.messages],
        sampling_params="testing",
        logprobs=None,
        frequency_penalty=None,
        stream=request.stream,
        # Groq only supports n=1 at the time of writing
        n=1,
        max_tokens=request.sampling_params.max_tokens or None,
        # TODO - does the default temperature make sense?
        temperature=request.sampling_params.temperature,
        # TODO - does the default top_p make sense?
        top_p=request.sampling_params.top_p,
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
