from pydantic import BaseModel
import os
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

class Explanation(BaseModel):
    explanation: str

class StepByStepCompletionResponse(BaseModel):
    steps: list[Explanation]

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me a proof for the chain rule in calculus. Take me through it step by step."}
    ],
    response_format={
      "type": "json_schema",
      "json_schema": StepByStepCompletionResponse.model_json_schema()
    }
)
print(response.completion_message.content)