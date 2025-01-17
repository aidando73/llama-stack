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

# Together build from source
pip install -e . \
    && llama stack build --template together --image-type conda \
    && llama stack run ./distributions/together/run.yaml \
    --port 5000 \
    --env TOGETHER_API_KEY=$TOGETHER_API_KEY | tee -a llama-stack.log

# Setup
screen -S llama-stack
source ~/miniconda3/bin/activate && conda activate ./env

# Remote VLLM build from source
export INFERENCE_MODEL=unsloth/Llama-3.3-70B-Instruct-bnb-4bit && \
export VLLM_URL=http://localhost:8000 && \
pip install -e .[all] && \
llama stack build --template remote-vllm --image-type conda && \
llama stack run ./distributions/remote-vllm/run.yaml --port 5000

# Fireworks build from source
pip install -e . \
&& llama stack build --config distributions/fireworks/build.yaml --image-type conda \
&& stdbuf --output=L llama stack run distributions/fireworks/run.yaml \
  --port 5000 | tee -a llama-stack.log
```
