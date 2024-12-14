```bash
source ~/miniconda3/bin/activate
conda create --prefix ./envs python=3.10

source ~/miniconda3/bin/activate
conda activate ./envs

pip install -e . \
&& llama stack build --config ./build.yaml --image-type conda \
&& llama stack run ./run.yaml \
  --port 5001
```
