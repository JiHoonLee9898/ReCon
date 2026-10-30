#!/bin/bash
# One-shot environment setup.
#   bash setup.sh            # creates ./venv, installs pinned deps, downloads LLaDA-8B-Instruct, Dream-v0-Instruct-7B and the 4 datasets
# Requirements: Linux, one NVIDIA GPU (>= 24 GB; the paper used a single H100), driver with CUDA >= 12.4,
#               Python 3.10 available as `python3.10` (or set PYTHON=/path/to/python3.10), internet.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.10}
VENV=./venv
MODEL_DIR=./models/LLaDA-8B-Instruct
DREAM_DIR=./models/Dream-v0-Instruct-7B

# 1) virtual environment with the exact versions used for the paper runs (see env.yml of the original repo)
if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip >/dev/null
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.25.0 transformers==4.49.0 accelerate==1.4.0 datasets==3.3.2 \
            sentencepiece tqdm huggingface_hub

# 2) model weights (public; ~16 GB + ~15 GB).  Skipped if already present.
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  hf download GSAI-ML/LLaDA-8B-Instruct --local-dir "$MODEL_DIR" 2>/dev/null \
    || huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir "$MODEL_DIR"
fi
if [ ! -f "$DREAM_DIR/model-00004-of-00004.safetensors" ]; then
  hf download Dream-org/Dream-v0-Instruct-7B --local-dir "$DREAM_DIR" 2>/dev/null \
    || huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir "$DREAM_DIR"
fi

# 3) the four evaluation datasets (cached by `datasets`; skipped if already cached)
python - <<'EOF'
from datasets import load_dataset
load_dataset("gsm8k", "main", split="test")
load_dataset("ai2_arc", "ARC-Challenge", split="test")
load_dataset("EleutherAI/truthful_qa_mc", "multiple_choice", split="validation")
load_dataset("HuggingFaceH4/MATH-500", split="test")
print("datasets ready")
EOF

python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"
echo "setup done. Run:  bash run.sh"
