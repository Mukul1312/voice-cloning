#!/usr/bin/env bash
# =====================================================================================
# pod_setup.sh — one-time environment setup on a RunPod pod.
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 — CUDA 12.4 floor, so it starts
# on any host with driver >= 12.4 (the cu128/torch280 image needs a 12.8 host and won't start on 12.7).
# pyannote.audio 4.0 PINS torch==2.8.0, so we install the torch trio FRESH from the cu126 index
# (torch 2.8 has no cu124 wheels; the cu126 build runs on any host driver >= 12.6). torchaudio 2.8 +
# torchcodec 0.7 is the pyannote-safe set (torchaudio>=2.9 removes AudioMetaData; torchcodec>=0.8 segfaults).
# Usage:  bash cloud/pod_setup.sh
# =====================================================================================
set -euo pipefail

echo "[setup 1/6] apt: ffmpeg + git + C/C++ toolchain (ctc-forced-aligner builds a pybind11 ext from source) ..."
apt-get -qq update && apt-get -qq install -y ffmpeg git build-essential python3-dev >/dev/null

echo "[setup 2/6] numpy pin (keeps numba/librosa coherent) ..."
pip -q install "numpy>=2.1,<2.3"

echo "[setup 3/6] torch 2.8 trio, cu126 build (replaces image torch 2.4; runs on host CUDA>=12.6; pyannote-4.0 pins torch 2.8) ..."
pip -q install --index-url https://download.pytorch.org/whl/cu126 "torch==2.8.0" "torchaudio==2.8.0" "torchcodec==0.7"

echo "[setup 4/6] diarization + speaker-verify ..."
pip -q install "pyannote.audio>=4.0,<5.0" "speechbrain>=1.0.0"

echo "[setup 5/6] forced aligner (MMS, --no-deps so it can't float torch/torchcodec) + ASR-QA ..."
pip -q install --no-deps git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
pip -q install uroman nltk "transformers>=4.48" Unidecode
pip -q install "faster-whisper==1.2.0" "ctranslate2>=4.5" "jiwer==4.0.0"

echo "[setup 6/6] audio I/O + QA metrics ..."
pip -q install "librosa>=0.11" soundfile

echo ""
echo "[verify] expose torch's cuDNN-9 libs (ctranslate2/faster-whisper need them at load) ..."
CUDNN_DIR="$(python -c 'import torch,glob,os;d=sorted({os.path.dirname(p) for p in glob.glob(os.path.join(os.path.dirname(torch.__file__),"..","nvidia","cudnn","lib","*.so*"))});print(d[0] if d else "")')"
if [ -n "$CUDNN_DIR" ]; then
  export LD_LIBRARY_PATH="$CUDNN_DIR:${LD_LIBRARY_PATH:-}"
  echo "export LD_LIBRARY_PATH=\"$CUDNN_DIR:\$LD_LIBRARY_PATH\"" > /etc/profile.d/cudnn.sh   # persist for future shells
  echo "  cuDNN dir: $CUDNN_DIR"
fi

echo "[verify] imports + GPU + cuDNN-9 Whisper smoke test (fails setup, not a paid run, on a broken stack) ..."
python - <<'PY'
import torch, torchaudio, torchcodec
assert torch.version.cuda,        "torch lost its CUDA build (clobbered by a PyPI wheel) — reinstall on the cu128 index"
assert torch.cuda.is_available(), "CUDA not available — torch was clobbered or no GPU attached"
assert torch.__version__.startswith("2.8."), f"torch {torch.__version__} != 2.8.x (pyannote-4.0 needs 2.8)"
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__, "| torchcodec", torchcodec.__version__,
      "| numpy", __import__("numpy").__version__, "| gpu", torch.cuda.get_device_name(0))
import pyannote.audio, speechbrain, ctc_forced_aligner, faster_whisper, jiwer, librosa
print("pipeline imports OK: pyannote", pyannote.audio.__version__, "| speechbrain", speechbrain.__version__)
# cuDNN-9 smoke test: a CUDA WhisperModel forces the ctranslate2 cuDNN load NOW (float16 = what Stage F uses).
from faster_whisper import WhisperModel
_m = WhisperModel("tiny", device="cuda", compute_type="float16"); del _m
print("faster-whisper/ctranslate2 CUDA load OK (cuDNN-9 resolved).")
PY
echo ""
echo "[setup] DONE. Next: export HF_TOKEN=...  then  python cloud/pod_dataprep.py --slug <slug> --ref <ref.wav>"
