"""
infer_curve.py — generate the learning-curve eval matrix on the pod.
For each subset n{N} and a sweep of checkpoints, generate the UNSEEN sentences x seeds (same prompt /
denoise / sentences / seeds as every prior eval, so it's apples-to-apples). Names:
  n{N}_k{step}__{sent}__s{seed}.wav   -> scored locally by scripts/score_curve.py
Resumable. Run on the pod after the 4 subset trainings (pod_curve.sh does this).
"""
import gc
import json
import os
import sys
import torch
import soundfile as sf
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

BASE = "/workspace/VoxCPM2"
REF = "/workspace/voice-cloning/data/out/lecture1/final/clips/ggs_l1_0112.wav"
REF_TEXT = ("Namabhasa means offences are not completely gone. If offences will completely go, "
            "then pure name will rise, sun will arise, moon will rise, then prema will come")
OUT = "/workspace/out_curve"
SUBSETS = [35, 70, 105, 140]
STEPS = [25, 50, 100, 200]                  # sweep -> pick each subset's best checkpoint locally
SENTS = {
    "nv1": "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death.",   # UNSEEN
    "nv2": "The pure devotee always remembers Krishna and never forgets Him for a single moment.",      # UNSEEN
}
SEEDS = [42, 123, 7]
os.makedirs(OUT, exist_ok=True)


def run(N, step):
    lora_dir = f"/workspace/ckpt/curve_n{N}/step_{step:07d}"
    if not os.path.isdir(lora_dir):
        print(f"skip n{N} step {step} (missing {lora_dir})", file=sys.stderr)
        return
    pending = [(sid, s) for sid in SENTS for s in SEEDS
               if not os.path.exists(f"{OUT}/n{N}_k{step}__{sid}__s{s}.wav")]
    if not pending:
        return
    info = json.load(open(f"{lora_dir}/lora_config.json"))
    cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
    print(f"=== n{N} step {step} ===", file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                               lora_config=cfg, lora_weights_path=lora_dir)
    sr = m.tts_model.sample_rate
    for sid, s in pending:
        wav = m.generate(text=SENTS[sid], prompt_wav_path=REF, prompt_text=REF_TEXT, denoise=True, seed=s)
        sf.write(f"{OUT}/n{N}_k{step}__{sid}__s{s}.wav", wav, sr)
        print(f"wrote n{N}_k{step}__{sid}__s{s}.wav", file=sys.stderr)
    del m; gc.collect(); torch.cuda.empty_cache()


for N in SUBSETS:
    for step in STEPS:
        run(N, step)
print("ALL DONE")
