#!/usr/bin/env bash
# =====================================================================================
# pod_denoise.sh — clean the 140 GGS training clips, then prep them for a clean retrain.
#
# THE EXPERIMENT: the only untried, high-leverage lever (confirmed across 6 community
# VoxCPM fine-tune repos — NONE denoise their training audio). We denoise the existing
# 140 clips and retrain with an IDENTICAL config -> a single-variable test of whether
# "noisy training data" is what caps us at ECAPA ~0.74.
#
# IDENTITY-SAFE BY DESIGN: we use `--denoise_only` (the denoiser, which separates speech
# from noise). We deliberately DO NOT run the full enhancer — it is GENERATIVE (bandwidth
# extension + distortion restoration) and can shift timbre, which is the very thing we clone.
# Source: github.com/resemble-ai/resemble-enhance (denoiser vs enhancer modules).
#
#   bash cloud/pod_denoise.sh
# Then run the identity safety-gate (scripts/ecapa_denoise_check.py) BEFORE retraining.
# =====================================================================================
set -e
cd /workspace
REPO=/workspace/voice-cloning
CLIPS=$REPO/data/out/lecture1/final/clips
DEN44=$REPO/data/out/lecture1/final/clips_dn_44k
OUT16=$REPO/data/out/lecture1/final/clips_dn
OUTMAN=$REPO/data/out/lecture1/final/train_dn.voxcpm.jsonl
# prefer the manifest the trainer actually used; fall back to the canonical text manifest
SRCMAN=$REPO/data/out/lecture1/final/train.voxcpm.jsonl
[ -f "$SRCMAN" ] || SRCMAN=$REPO/data/out/lecture1/final/train.jsonl

echo "=== 1) venv for resemble-enhance (kept separate from train/dataprep envs) ==="
python3.11 -m venv venv_dn 2>/dev/null || python3 -m venv venv_dn
source venv_dn/bin/activate
pip install -q -U pip
# resemble-enhance pins older deps; a fresh venv avoids clashing with the torch-2.5 train env.
pip install -q resemble-enhance --upgrade
pip install -q soundfile scipy

echo "=== 2) DENOISE-ONLY  $CLIPS -> $DEN44  (44.1kHz out, GPU auto-detected) ==="
resemble-enhance "$CLIPS" "$DEN44" --denoise_only

echo "=== 3) post-process -> 16k + peak-norm(0.95) + trailing-silence trim + manifest ==="
python "$REPO/cloud/denoise_postproc.py" \
  --src_manifest "$SRCMAN" \
  --denoised_dir "$DEN44" \
  --out_clips "$OUT16" \
  --out_manifest "$OUTMAN"

echo ""
echo "=== NEXT: identity safety-gate (do NOT skip) ==="
echo "  python scripts/ecapa_denoise_check.py   # denoised clips must still match HIS centroid"
echo "=== THEN retrain (single variable changed = denoised data): ==="
echo "  bash cloud/pod_train.sh  with  --config_path cloud/voxcpm_lora_dn.yaml"
echo "=== done ==="
