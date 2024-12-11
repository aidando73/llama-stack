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
    ToolCallDelta,
    ToolCall,
    ToolCallParseStatus,
)
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_models.sku_list import CoreModelId
from llama_models.llama3.api.datatypes import StopReason
from groq import Groq
from groq.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
import json

class GroqInferenceAdapter(Inference):
    def __init__(self, config: GroqConfig):
        pass