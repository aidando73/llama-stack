
import json

with open("messages.json", "r") as f:
    messages = json.load(f)

print(messages["messages"][0]["content"]["text"])
