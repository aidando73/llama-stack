from pydantic import BaseModel
import json
from openai import OpenAI

class CompletionResponse(BaseModel):
    completion_message: str

client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake")

completion = client.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    prompt="Write me a print statement that prints 'Hello, World!' in python",
    extra_body={
        "guided_json": CompletionResponse.model_json_schema(),
    },
)

print(completion.choices[0].text)
