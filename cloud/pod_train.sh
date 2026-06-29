#!/usr/bin/env bash
# =====================================================================================
# pod_train.sh — VoxCPM2 LoRA training env + run, on a RunPod RTX 4090 (24GB, CUDA 12.4 host).
#
# Training needs torch >=2.5 (the dataprep env was torch 2.4) -> we make a SEPARATE venv.
# torch 2.5 cu124 wheels run on a 12.4 host driver. Verify VoxCPM's repo script paths on first
# run; if the repo layout differs from the docs, adjust step 6.
#   bash cloud/pod_train.sh
# =====================================================================================
set -e
cd /workspace

echo "=== 1) fresh training venv (torch 2.5+, separate from dataprep) ==="
python3.11 -m venv venv_train 2>/dev/null || python3 -m venv venv_train
source venv_train/bin/activate
pip install -q -U pip

echo "=== 2) torch 2.5 (cu124 — runs on the 12.4 host driver) ==="
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== 3) VoxCPM + training deps ==="
[ -d VoxCPM ] || git clone https://github.com/OpenBMB/VoxCPM.git
pip install -q -e VoxCPM || pip install -q -r VoxCPM/requirements.txt
pip install -q tensorboardX argbind transformers librosa safetensors "huggingface_hub[cli]"

echo "=== 4) base model openbmb/VoxCPM2 (~several GB; set HF_TOKEN if it 403s) ==="
huggingface-cli download openbmb/VoxCPM2 --local-dir /workspace/VoxCPM2

echo "=== 5) sanity: torch CUDA + data present ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
n=sum(1 for _ in open("/workspace/voice-cloning/data/out/lecture1/final/train.voxcpm.jsonl"))
print("train samples:", n)
PY

echo "=== 6) train (LoRA r=32; checkpoints -> /workspace/ckpt/lora every 50 steps) ==="
mkdir -p /workspace/ckpt/lora /workspace/logs/lora
python VoxCPM/scripts/train_voxcpm_finetune.py \
    --config_path /workspace/voice-cloning/cloud/voxcpm_lora.yaml

echo "=== done. evaluate checkpoints with cloud/TRAINING.md step 5 ==="
