```bash
export INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
# ollama names this model differently, and we must use the ollama name when loading the model
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
export LLAMA_STACK_PORT=5001
ollama run $OLLAMA_INFERENCE_MODEL --keepalive 60m

docker run \
  -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434



source ~/miniconda3/bin/activate
conda create --prefix envs python=3.10

source ~/miniconda3/bin/activate
conda activate ./envs

# Run from source
llama stack build --template ollama --image-type conda
llama stack run ./distributions/ollama/run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434

pip install .

llama-stack-client --endpoint http://localhost:$LLAMA_STACK_PORT \
  inference chat-completion \
  --message "hello, what model are you?"
```