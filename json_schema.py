import os
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

models = client.models.list()
print(models)

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    response_format={
      "type": "json_schema",
      "json_schema": {
        "type": "object",
        "properties": {
          "completion_message": {
            "type": "object",
            "properties": {
              "content": {"type": "string"},
              "additional_info": {"type": "string"}
            }
          }
        }
      }
    }
)
print(response.completion_message.content)