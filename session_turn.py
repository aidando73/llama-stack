# Stored in agents_store.db
# turn_id => session:f4920b89-1035-4432-92ab-3d800878e28d:7b19e203-53cc-4295-b6cf-f0c400611ed1:e38da75e-70fb-4895-b522-b25373f3e8d5
format = {
    "turn_id": "e38da75e-70fb-4895-b522-b25373f3e8d5",
    "session_id": "7b19e203-53cc-4295-b6cf-f0c400611ed1",
    "input_messages": [
        {
            "role": "user",
            "content": "Which family of models was released on October 24, 2024?",
            "context": null,
        },
        {
            "role": "user",
            "content": "What about Llama 3.1 family of models, what is the release date for it?",
            "context": "Here are the retrieved documents for relevant context:\n=== START-RETRIEVED-CONTEXT ===\n\nid:llama_3.1.md; content: for improved inference scalability.\n\nModel Release Date: July 23, 2024.\n\nStatus: This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model safety with community feedback.\n\nLicense: A custom commercial license, the Llama 3.1 Community License, is available at: [https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE](https://github.com/meta-\nid:llama_3.1.md; content:_1/LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)\n\nFeedback: Instructions on how to provide feedback or comments on the model can be found in the Llama Models [README](https://github.com/meta-llama/llama-models/blob/main/README.md). For more technical information about generation parameters and recipes for how to use Llama 3.1 in applications, please go [here](https://\nid:llama_3.2.md; content:.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) (a custom, commercial license agreement).\n\nFeedback: Instructions on how to provide feedback or comments on the model can be found in the Llama Models [README](https://github.com/meta-llama/llama-models/blob/main/README.md). For more technical information about generation parameters and recipes for how to use Llama 3.2 in applications, please go [here](https\n\n=== END-RETRIEVED-CONTEXT ===\n",
        },
    ],
    "steps": [
        {
            "turn_id": "e38da75e-70fb-4895-b522-b25373f3e8d5",
            "step_id": "71483ae0-fb77-4dbd-8ac0-d65210547101",
            "started_at": null,
            "completed_at": null,
            "step_type": "memory_retrieval",
            "memory_bank_ids": ["test_bank_2"],
            "inserted_context": [
                "Here are the retrieved documents for relevant context:\n=== START-RETRIEVED-CONTEXT ===\n",
                "id:llama_3.1.md; content: for improved inference scalability.\n\nModel Release Date: July 23, 2024.\n\nStatus: This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model safety with community feedback.\n\nLicense: A custom commercial license, the Llama 3.1 Community License, is available at: [https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE](https://github.com/meta-",
                "id:llama_3.1.md; content:_1/LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)\n\nFeedback: Instructions on how to provide feedback or comments on the model can be found in the Llama Models [README](https://github.com/meta-llama/llama-models/blob/main/README.md). For more technical information about generation parameters and recipes for how to use Llama 3.1 in applications, please go [here](https://",
                "id:llama_3.2.md; content:.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) (a custom, commercial license agreement).\n\nFeedback: Instructions on how to provide feedback or comments on the model can be found in the Llama Models [README](https://github.com/meta-llama/llama-models/blob/main/README.md). For more technical information about generation parameters and recipes for how to use Llama 3.2 in applications, please go [here](https",
                "\n=== END-RETRIEVED-CONTEXT ===\n",
            ],
        },
        {
            "turn_id": "e38da75e-70fb-4895-b522-b25373f3e8d5",
            "step_id": "c5215e57-57eb-44fb-b800-5fa0d712c4ab",
            "started_at": null,
            "completed_at": null,
            "step_type": "inference",
            "model_response": {
                "role": "assistant",
                "content": "The Llama 3.1 family of models was released on July 23, 2024.",
                "stop_reason": "end_of_turn",
                "tool_calls": [],
            },
        },
    ],
    "output_message": {
        "role": "assistant",
        "content": "The Llama 3.1 family of models was released on July 23, 2024.",
        "stop_reason": "end_of_turn",
        "tool_calls": [],
    },
    "output_attachments": [],
    "started_at": "2024-11-29T23:43:10.133055",
    "completed_at": "2024-11-29T23:43:11.406677",
}
