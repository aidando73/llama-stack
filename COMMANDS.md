```bash
# Environment setup
source ~/miniconda3/bin/activate && conda activate ./env

screen -S llama-stack

# Fireworks from docker
export LLAMA_STACK_PORT=5000
docker run -it \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama \
  llamastack/distribution-fireworks \
  --port $LLAMA_STACK_PORT \
  --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY

# Ollama build from source
export LLAMA_STACK_PORT=5000
pip install -e . \
    && llama stack build --template ollama --image-type conda \
    && llama stack run ./distributions/ollama/run.yaml \
    --port $LLAMA_STACK_PORT \
    --env OLLAMA_URL=http://localhost:11434


# Setup
source ~/miniconda3/bin/activate && conda activate ./env
mkfifo llama-pipe
screen -S llama-stack

# Fireworks build from source
pip install -e . \
&& llama stack build --config distributions/fireworks/build.yaml --image-type conda \
&& llama stack run distributions/fireworks/run.yaml \
  --port 5000 \
  | stdbuf -o0 tee -a llama-pipe
```
