#!/usr/bin/env bash
# =====================================================================================
# pod_curve.sh — the DATA-QUANTITY LEARNING CURVE on one pod.
# Trains the LoRA on 4 nested subsets (n35/n70/n105/n140 of the SAME 140 clips), sweeping
# checkpoints, then generates the unseen eval matrix. Scored locally by scripts/score_curve.py.
# Reads whether best-unseen-ECAPA rises with data size -> is "quantity" a real lever or not.
#
#   bash cloud/pod_curve.sh        (run detached; survives SSH drops)
# =====================================================================================
set -e
cd /workspace
REPO=/workspace/voice-cloning

echo "=== [curve] 1) env setup + train smallest subset (pod_train.sh: venv/torch/VoxCPM/base/memfix) ==="
bash "$REPO/cloud/pod_train.sh" "$REPO/cloud/voxcpm_lora_n35.yaml"

echo "=== [curve] 2) train remaining subsets (env already set up) ==="
source venv_train/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for N in 70 105 140; do
  echo "=== [curve] training subset n$N ==="
  python VoxCPM/scripts/train_voxcpm_finetune.py --config_path "$REPO/cloud/voxcpm_lora_n${N}.yaml"
done

echo "=== [curve] 3) eval sweep (subsets x checkpoints x unseen x seeds) ==="
python "$REPO/cloud/infer_curve.py"

echo "=== [curve] checkpoints present ==="
for N in 35 70 105 140; do echo -n "n$N: "; ls -d /workspace/ckpt/curve_n$N/step_* 2>/dev/null | sed 's#.*/##' | tr '\n' ' '; echo; done
echo "=== CURVE DONE ==="
