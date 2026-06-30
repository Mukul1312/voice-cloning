"""
infer_compare.py — 250 vs 350 head-to-head, with an overfitting guard.
Generates, with the denoised ggs_l1_0112 prompt:
  - "nm" = the Namabhasa line (SEEN — it's a training clip; lets us A/B vs his real audio)
  - "nv" = a NOVEL devotional line (UNSEEN — the clean generalization test, multi-seed)
for both the step-250 and step-350 LoRA. Outputs -> /workspace/out_cmp/
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
NM = ("Namabhasa means offences are not completely gone. If offences will completely go, "
      "then pure name will rise")                                    # SEEN (training content)
NV = "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death."  # UNSEEN
OUT = "/workspace/out_cmp"
JOBS = [("nm", NM, [42]), ("nv", NV, [42, 123])]
os.makedirs(OUT, exist_ok=True)


def run(lora_dir, tag):
    info = json.load(open(f"{lora_dir}/lora_config.json"))
    cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
    print(f"loading {tag} ({lora_dir})...", file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                               lora_config=cfg, lora_weights_path=lora_dir)
    sr = m.tts_model.sample_rate
    for tname, text, seeds in JOBS:
        for s in seeds:
            wav = m.generate(text=text, prompt_wav_path=REF, prompt_text=REF_TEXT, denoise=True, seed=s)
            o = f"{OUT}/{tag}_{tname}_s{s}.wav"
            sf.write(o, wav, sr)
            print(f"wrote {o}  ({len(wav)/sr:.2f}s)", file=sys.stderr)
    del m; gc.collect(); torch.cuda.empty_cache()


run("/workspace/lora250", "s250")
run("/workspace/lora350", "s350")
print("ALL DONE")
