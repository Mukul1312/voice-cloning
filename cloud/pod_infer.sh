#!/usr/bin/env bash
# =====================================================================================
# pod_infer.sh — inference-only env for the GGS clone (denoised-reference prompted gen).
# Fresh RunPod RTX 4090 (cu124 host). No training -> from_pretrained loads bf16, no memfix needed.
#   bash cloud/pod_infer.sh   (the step-250 LoRA is uploaded separately to /workspace/lora250)
# =====================================================================================
set -e
cd /workspace

echo "=== 1) venv + torch 2.5 (cu124, runs on the host driver) ==="
python3.11 -m venv venv 2>/dev/null || python3 -m venv venv
source venv/bin/activate
pip install -q -U pip
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== 2) VoxCPM + deps (modelscope pulls the zipenhancer denoiser on first use) ==="
[ -d VoxCPM ] || git clone https://github.com/OpenBMB/VoxCPM.git
pip install -q -e VoxCPM || pip install -q -r VoxCPM/requirements.txt
pip install -q "huggingface_hub[cli]" soundfile librosa modelscope

echo "=== 3) base model openbmb/VoxCPM2 (~several GB) ==="
hf download openbmb/VoxCPM2 --local-dir /workspace/VoxCPM2

echo "=== 4) repo (reference clips + texts) ==="
[ -d voice-cloning ] || git clone https://github.com/Mukul1312/voice-cloning.git

echo "=== 5) sanity ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
echo "=== SETUP DONE — LoRA expected at /workspace/lora250 ==="
