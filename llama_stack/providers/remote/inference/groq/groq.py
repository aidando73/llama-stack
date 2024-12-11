import warnings
from typing import AsyncIterator, List, Optional, Union, AsyncGenerator
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
)
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_models.sku_list import CoreModelId
from llama_models.llama3.api.datatypes import StopReason
from groq import Groq

def _convert_groq_tool_definition(tool_definition: ToolDefinition) -> dict:
    tool_parameters = tool_definition.parameters or {}
    # TODO - use groq types
    return {
        # Groq only supports function tools are supported at this stage
        "type": "function",
        "function": {
            "name": tool_definition.tool_name,
            "description": tool_definition.description,
            "parameters": [
                _convert_groq_tool_parameter(key, param) for key, param in tool_parameters.items()
            ],
        },
    }
def _convert_groq_tool_parameter(name: str, tool_parameter: ToolParamDefinition) -> dict:
    # TODO - use groq types
    return {
        "name": name,
        "type": tool_parameter.param_type,
        "description": tool_parameter.description,
        "required": tool_parameter.required,
        "default": tool_parameter.default,
    }

_MODEL_ALIASES = [
    build_model_alias(
        "llama3-8b-8192",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
]

class GroqInferenceAdapter(Inference, ModelRegistryHelper):
    client: Groq
    def __init__(self, config: GroqConfig) -> None:
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
        raise NotImplementedError()

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
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
    ) -> Union[
        ChatCompletionResponse, AsyncGenerator
    ]:
        if logprobs:
            warnings.warn("logprobs are not supported yet")
        
        if response_format:
            warnings.warn("response_format is not supported yet")

        if sampling_params.repetition_penalty:
            warnings.warn("repetition_penalty is not supported")

        if sampling_params.strategy != SamplingStrategy.greedy:
            warnings.warn("sampling_params.strategy is not supported")
        
        if stream:
            response = self._client.chat.completions.create(
                model=self.get_provider_model_id(model_id),
                messages=messages,
                stream=True,
                # Groq doesn't support logprobs yet
                logprobs=False,
                # Groq only supports n=1 at this stage
                n=1,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                # Assume that a max_tokens of 0 means client has not set a value
                max_tokens=sampling_params.max_tokens or None,
                # frequency_penalty and sampling_params.repetition_penalty seem to have different semantics
                # frequency_penalty defaults to 0 is a float between -2.0 and 2.0
                # repetition_penalty defaults to 1 and is often set somewhere between 1.0 and 2.0
                # so we exclude it for now
                frequency_penalty=None,
                tools=tools
            )
            async def stream_response():
                for chunk in response:
                    if (chunk.choices[0].finish_reason == 'stop'
                        or chunk.choices[0].finish_reason == 'length'):
                        if chunk.choices[0].finish_reason == 'length':
                            stop_reason = StopReason.out_of_tokens
                        elif chunk.choices[0].finish_reason == 'stop':
                            stop_reason = StopReason.end_of_message
                        else:
                            warnings.warn(f"Unknown finish reason: {chunk.choices[0].finish_reason}")
                            stop_reason = StopReason.end_of_message
                        yield ChatCompletionResponseStreamChunk(
                            event=ChatCompletionResponseEvent(
                                event_type=ChatCompletionResponseEventType.complete,
                                delta="",
                                stop_reason=stop_reason,
                            )
                        )
                        return
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=chunk.choices[0].delta.content,
                    )
                )
            return stream_response()

        print(self.get_provider_model_id(model_id))
        response = self.client.chat.completions.create(
            model=self.get_provider_model_id(model_id),
            messages=messages,
        )
        return ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=response.choices[0].message.content,
                stop_reason=StopReason.end_of_turn,
            ),
            logprobs=None,
        )
        # async def generate_alphabet():
        #     for letter in 'abcdefghijklmnopqrstuvwxyz':
        #         await asyncio.sleep(1)
        #         if letter == 'a':
        #             event_type = ChatCompletionResponseEventType.start
        #         elif letter == 'z':
        #             event_type = ChatCompletionResponseEventType.complete
        #         else:
        #             event_type = ChatCompletionResponseEventType.progress
        #         yield ChatCompletionResponseStreamChunk(
        #             event=ChatCompletionResponseEvent(
        #                 event_type=event_type,
        #                 delta=letter,
        #                 stop_reason=StopReason.end_of_turn if letter == 'z' else None
        #             )
        #         )
        # return generate_alphabet()