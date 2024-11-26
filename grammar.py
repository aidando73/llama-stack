import os
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:5001")

models = client.models.list()
print(models)

response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding"}
    ],
    response_format={
        "type": "grammar",
        "bnf": {
            "<start>": "<haiku>",
            "<haiku>": "<line> <line> <line>",
            "<line>": "<noun> <verb> <noun>",
            "<noun>": "code | coding | computer",
            "<verb>": "writes | writes | types"
        }
        # "bnf": """
        #   <start> ::= <haiku>
        #   <haiku> ::= <line> <line> <line>
        #   <line> ::= <noun> <verb> <noun>
        #   <noun> ::= "code" | "coding" | "computer"
        #   <verb> ::= "writes" | "writes" | "types"
        # """
    }
)
print(response.completion_message.content)
