
import pytest

from llama_stack.providers.remote.inference.ollama import OllamaImplConfig

class TestGroqInit:
    def test_raises_runtime_error_if_config_is_not_groq_config(self):
        from llama_stack.providers.remote.inference.groq import get_adapter_impl
        config = OllamaImplConfig(model="llama3.1-8b-8192")
        with pytest.raises(RuntimeError):
            get_adapter_impl(config, None)
