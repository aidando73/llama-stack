# Stored in agents_store.db
# agent_id => agent:f4920b89-1035-4432-92ab-3d800878e28d
agent = {
    "sampling_params": {
        "strategy": "greedy",
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 0,
        "max_tokens": 0,
        "repetition_penalty": 1.0,
    },
    "input_shields": [],
    "output_shields": [],
    "tools": [
        {
            "input_shields": [],
            "output_shields": [],
            "type": "memory",
            "memory_bank_configs": [{"bank_id": "test_bank_2", "type": "vector"}],
            "query_generator_config": {"type": "default", "sep": " "},
            "max_tokens_in_context": 300,
            "max_chunks": 5,
        }
    ],
    "tool_choice": "auto",
    "tool_prompt_format": "json",
    "max_infer_iters": 10,
    "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "instructions": "You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
    "enable_session_persistence": true,
}
