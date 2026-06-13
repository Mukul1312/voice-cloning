#!/usr/bin/env bash
# =====================================================================================
# pod_setup.sh — one-time env setup on a RunPod pod (the torch-2.4 / CUDA-12.4 host path).
#
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04  (template id runpod-torch-v240)
#   -> CUDA 12.4 floor: starts on any 4090 host with driver >= 12.4. (The cu128/torch280 image needs a
#      12.8 host driver and FAILS to start on common 12.4/12.7 hosts — verified the hard way.)
#
# We KEEP the image's torch 2.4.1+cu124 / torchaudio 2.4.1 (run fine on a 12.4 driver) and add pyannote 3.x
# (pyannote 4.0 needs torch 2.8 -> a >=12.6 host we can't guarantee). The tricky part is a 3-way version
# deadlock: pyannote 3.4 needs huggingface_hub<1.0, but the ctc-aligner needs transformers>=4.48 which
# needs hub>=1.0. We resolve it by pinning transformers 4.46 + hub 0.26 (pyannote-compatible) and patching
# the aligner's single incompatible kwarg (dtype -> torch_dtype, which transformers 4.46 still uses).
# Versions below are the exact working set captured from a green run.
# Usage:  bash cloud/pod_setup.sh
# =====================================================================================
set -euo pipefail

echo "[setup 1/5] apt: ffmpeg + git + C/C++ toolchain (ctc-aligner builds a pybind11 ext from source) ..."
apt-get -qq update && apt-get -qq install -y ffmpeg git build-essential python3-dev >/dev/null

echo "[setup 2/5] diarization (pyannote 3.x — torch-2.4 compatible) + ECAPA speaker-verify ..."
pip -q install "pyannote.audio==3.4.0" "speechbrain==1.1.0"

echo "[setup 3/5] forced aligner (MMS) --no-deps + the version-locked set that reconciles pyannote+aligner ..."
pip -q install --no-deps git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
pip -q install uroman nltk Unidecode "transformers==4.46.3" "huggingface_hub==0.26.5" "tokenizers==0.20.3"
# The installed aligner calls AutoModelForCTC.from_pretrained(model_path, dtype=...) — a kwarg only added
# in transformers>=4.48. Rename it to torch_dtype so it works on the pinned 4.46. (The aligner already
# loads audio via torchaudio, not torchcodec, so torchcodec is NOT needed on torch 2.4.)
ALN=$(python -c "import ctc_forced_aligner,os;print(os.path.join(os.path.dirname(ctc_forced_aligner.__file__),'alignment_utils.py'))")
sed -i 's/from_pretrained(model_path, dtype=dtype)/from_pretrained(model_path, torch_dtype=dtype)/' "$ALN"
echo "  patched aligner dtype kwarg in $ALN"

echo "[setup 4/5] ASR round-trip QA (faster-whisper + cuDNN-9 ctranslate2) + CER metrics ..."
pip -q install "faster-whisper==1.2.0" "ctranslate2>=4.5" "jiwer==4.0.0"

echo "[setup 5/5] audio I/O ..."
pip -q install "librosa>=0.11" soundfile

echo ""
echo "[verify] expose torch's bundled cuDNN-9 libs (ctranslate2/faster-whisper need them at load) ..."
CUDNN_DIR="$(python -c 'import torch,glob,os;d=sorted({os.path.dirname(p) for p in glob.glob(os.path.join(os.path.dirname(torch.__file__),"..","nvidia","cudnn","lib","*.so*"))});print(d[0] if d else "")')"
if [ -n "$CUDNN_DIR" ]; then
  export LD_LIBRARY_PATH="$CUDNN_DIR:${LD_LIBRARY_PATH:-}"
  echo "export LD_LIBRARY_PATH=\"$CUDNN_DIR:\$LD_LIBRARY_PATH\"" > /etc/profile.d/cudnn.sh   # persist for future shells
  echo "  cuDNN dir: $CUDNN_DIR"
fi

echo "[verify] imports + GPU + aligner-model + cuDNN Whisper smoke test (fails setup, not a paid run) ..."
python - <<'PY'
import torch, torchaudio
assert torch.cuda.is_available(), "no CUDA GPU attached"
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| gpu", torch.cuda.get_device_name(0))
import pyannote.audio, speechbrain, ctc_forced_aligner, faster_whisper, jiwer, librosa, transformers, huggingface_hub
print("imports OK: pyannote", pyannote.audio.__version__, "| transformers", transformers.__version__,
      "| hub", huggingface_hub.__version__, "| speechbrain", speechbrain.__version__)
from ctc_forced_aligner import load_alignment_model           # forces the dtype-patched model load
m, tok = load_alignment_model("cuda", dtype=torch.float16); del m
print("aligner (MMS) model load OK")
from faster_whisper import WhisperModel                        # forces the ctranslate2 cuDNN-9 load NOW
w = WhisperModel("tiny", device="cuda", compute_type="float16"); del w
print("faster-whisper/ctranslate2 CUDA load OK (cuDNN-9 resolved).")
PY
echo ""
echo "[setup] DONE. Next: write your HF token ->  echo 'hf_xxx' > /workspace/.hf_token"
echo "  then:  python cloud/pod_dataprep.py --slug <slug> --ref data/refs/<ref>.wav"
echo "  (accept the pyannote speaker-diarization-3.1 + segmentation-3.0 gated repos on HF first.)"
