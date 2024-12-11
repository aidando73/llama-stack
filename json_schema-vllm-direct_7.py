from pydantic import BaseModel
import json
from openai import OpenAI

class CompletionResponse(BaseModel):
    completion_message: str

client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake")

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Write me a haiku about coding",
        },
    ],
    extra_body={
        "guided_grammar": """
            "<start>": "<haiku>",
            "<haiku>": "<line> <line> <line>",
            "<line>": "<noun> <verb> <noun>",
            "<noun>": "code | coding | computer",
            "<verb>": "writes | writes | types"
"""
    },
)

print(completion.choices[0].message.content)