```bash

# Using Conda now
python -m venv .venv
source $STORAGE_DIR/llama-stack/.venv/bin/activate

conda create --name llama-stack python=3.10



source ~/miniconda3/bin/activate && conda activate llama-stack

pip install -e .

huggingface-cli login

# Env vars:
export OLLAMA_INFERENCE_MODEL="llama3.2:3b-instruct-fp16"
export LLAMA_STACK_PORT=5001
export INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct
export INFERENCE_PORT=8000
export VLLM_URL=http://localhost:8000/v1

# Remote vLLM
sudo docker run \
  -it \
  --net=host \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ./run.yaml:/root/my-run.yaml \
  llamastack/distribution-remote-vllm:0.0.54 \
  --yaml-config /root/my-run.yaml \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env VLLM_URL=http://localhost:$INFERENCE_PORT/v1

llama model download --model-id meta-llama/Llama-3.2-3B-Instruct
# Add in signed URL from email

# Meta reference gpu server
sudo docker run \
  -it \
  -v ~/.llama:/root/.llama \
  --gpus all \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  llamastack/distribution-meta-reference-gpu \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

# Fireworks server
sudo docker run \
    -it \
    -v ~/run.yaml:/root/run.yaml \
    --net=host \
    llamastack/distribution-fireworks \
    --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
    --env FIREWORKS_API_KEY=$FIREWORKS_API_KEY



llama-stack-client --endpoint http://localhost:$LLAMA_STACK_PORT   inference chat-completion   --message "hello, what model are you?"


# vLLM server
sudo docker run --gpus all \
    -v $STORAGE_DIR/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)" \
    -p 8000:$INFERENCE_PORT \
    --ipc=host \
    --net=host \
    vllm/vllm-openai:v0.6.3.post1 \
    --model $INFERENCE_MODEL


# Install the stack
llama stack build --template remote-vllm --image-type conda
# Run the stack
conda activate llamastack-remote-vllm
llama stack run run.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

llama stack build --template remote-vllm --image-type conda && llama stack run run.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=meta-llama/Llama-3.2-3B-Instruct

python json_schema.py
```