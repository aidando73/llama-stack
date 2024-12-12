import warnings
from typing import AsyncIterator, List, Optional, Union, AsyncGenerator
import json
from .groq_utils import (
    convert_chat_completion_request,
    convert_non_stream_chat_completion_response,
)
from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    Message,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
    SamplingStrategy,
    ToolParamDefinition,
)
from llama_models.datatypes import SamplingParams
from llama_models.sku_list import CoreModelId
from llama_models.llama3.api.datatypes import StopReason
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseStreamChunk,
    ChatCompletionResponseEventType,
    CompletionResponse,
    CompletionMessage,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    ResponseFormat,
    ToolCallDelta,
    ToolCall,
    ToolCallParseStatus,
)
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from groq import Groq
from groq.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

_MODEL_ALIASES = [
    build_model_alias(
        "llama3-8b-8192",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
]

class GroqInferenceAdapter(Inference, ModelRegistryHelper):
    _client: Groq

    def __init__(self, config: GroqConfig):
        ModelRegistryHelper.__init__(self, model_aliases=_MODEL_ALIASES)
        self._client = Groq(api_key=config.api_key)

    def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        # Groq doesn't support non-chat completion as of time of writing
        raise NotImplementedError()

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[
            ToolPromptFormat
        ] = None,  # API default is ToolPromptFormat.json, we default to None to detect user input
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncGenerator]:

        model_id = self.get_provider_model_id(model_id)
        request = convert_chat_completion_request(
            request=ChatCompletionRequest(
                model=model_id,
                messages=messages,
                sampling_params=sampling_params,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                tool_prompt_format=tool_prompt_format,
                stream=stream,
                logprobs=logprobs,
            )
        )

        response = self._client.chat.completions.create(**request)

        if stream:
            return None
        else:
            return convert_non_stream_chat_completion_response(response)

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
