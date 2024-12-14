# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import AsyncIterator, List, Optional, Union

from groq import Groq
from llama_models.datatypes import SamplingParams
from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    Message,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_models.sku_list import CoreModelId

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    ResponseFormat,
)
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    build_model_alias_with_just_provider_model_id,
    ModelRegistryHelper,
)
from .groq_utils import (
    convert_chat_completion_request,
    convert_chat_completion_response,
    convert_chat_completion_response_stream,
)

_MODEL_ALIASES = [
    build_model_alias(
        "llama3-8b-8192",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama-3.1-8b-instant",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "llama3-70b-8192",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_model_alias(
        "llama-3.3-70b-versatile",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    # Groq only contains a preview version for llama-3.2-3b
    # Preview models aren't recommended for production use, but we include this one
    # to pass the test fixture
    # TODO(aidand): Replace this with a stable model once Groq supports it
    build_model_alias(
        "llama-3.2-3b-preview",
        CoreModelId.llama3_2_3b_instruct.value,
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
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        model_id = self.get_provider_model_id(model_id)
        if model_id == "llama-3.2-3b-preview":
            warnings.warn(
                "Groq only contains a preview version for llama-3.2-3b-instruct. "
                "Preview models aren't recommended for production use. "
                "They can be discontinued on short notice."
            )

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
            return convert_chat_completion_response_stream(response)
        else:
            return convert_chat_completion_response(response)

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()