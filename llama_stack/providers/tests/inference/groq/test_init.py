import pytest

from llama_stack.providers.remote.inference.ollama import OllamaImplConfig
from llama_stack.providers.remote.inference.groq import get_adapter_impl
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.groq.groq import GroqInferenceAdapter
from llama_stack.apis.inference import Inference
import os


class TestGroqInit:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_if_config_is_not_groq_config(self):
        config = OllamaImplConfig(model="llama3.1-8b-8192")

        with pytest.raises(RuntimeError):
            await get_adapter_impl(config, None)

    @pytest.mark.asyncio
    async def test_returns_groq_adapter(self):
        config = GroqConfig()
        adapter = await get_adapter_impl(config, None)
        assert type(adapter) is GroqInferenceAdapter
        assert isinstance(adapter, Inference)

    def test_config_api_key_defaults_to_env_var(self):
        os.environ["GROQ_API_KEY"] = "test"
        config = GroqConfig()
        assert config.api_key == "test"
