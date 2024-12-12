import warnings
from typing import Literal, AsyncGenerator

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    Message,
    Role,
    ChatCompletionResponse,
    CompletionMessage,
    StopReason,
    ChatCompletionResponseStreamChunk,
)

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
from groq.types.chat.chat_completion import ChatCompletion
from groq.types.chat.chat_completion_chunk import ChatCompletionChunk
from groq import AsyncStream


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
        warnings.warn("repetition_penalty is not supported")
    
    if request.tools:
        warnings.warn("tools are not supported yet")

    return CompletionCreateParams(
        model=request.model,
        messages=[_convert_message(message) for message in request.messages],
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


def convert_non_stream_chat_completion_response(
    response: ChatCompletion,
) -> ChatCompletionResponse:
    # groq only supports n=1 at time of writing, so there is only one choice
    choice = response.choices[0]
    return ChatCompletionResponse(
        completion_message=CompletionMessage(
            content=choice.message.content,
            stop_reason=_map_finish_reason_to_stop_reason(choice.finish_reason),
        ),
    )

async def convert_stream_chat_completion_response(
    stream: AsyncStream[ChatCompletionChunk],
) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
    
    pass

def _map_finish_reason_to_stop_reason(finish_reason: Literal["stop", "length", "tool_calls"]) -> StopReason:
    """
    Convert a Groq chat completion finish_reason to a StopReason.

    finish_reason: Literal["stop", "length", "tool_calls"]
        - stop -> model hit a natural stop point or a provided stop sequence
        - length -> maximum number of tokens specified in the request was reached
        - tool_calls -> model called a tool
    """
    if finish_reason == "stop":
        return StopReason.end_of_turn
    elif finish_reason == "length":
        return StopReason.end_of_message
    elif finish_reason == "tool_calls":
        raise NotImplementedError("tool_calls is not supported yet")
    else:
        raise ValueError(f"Invalid finish reason: {finish_reason}")
