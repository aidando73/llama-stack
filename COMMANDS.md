```bash
source ~/miniconda3/bin/activate && conda activate ./envs

llama stack build --template ollama --image-type conda \
    && llama stack run ./distributions/ollama/run.yaml \
    --port $LLAMA_STACK_PORT \
    --env INFERENCE_MODEL=$INFERENCE_MODEL \
    --env OLLAMA_URL=http://localhost:11434
```