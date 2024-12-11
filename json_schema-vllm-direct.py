import json
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="fake"
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "user", "content": "Write me a print statement that prints 'Hello, World!' in python"},
    ],
    response_format={
      "type": "json_schema",
      "json_schema": {
          "name": "name_of_response_format",
          "schema": {
              "type": "object",
              "properties": {
                  "completion_message": {
                      "type": "string",
                  }
              },
                "required": ["completion_message"]
          },
          "strict": True
      }
    }
)

print(completion.choices[0].message.content)