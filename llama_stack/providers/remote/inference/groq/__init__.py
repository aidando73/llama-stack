
from llama_stack.apis.inference import Inference

from .config import GroqConfig


async def get_adapter_impl(config: GroqConfig, _deps) -> Inference:
    # import dynamically so `llama stack build` does not fail due to missing dependencies
    from .groq import GroqInferenceAdapter

    if not isinstance(config, GroqConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")

    adapter = GroqInferenceAdapter(config)
    return adapter