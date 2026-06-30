"""
infer_eval.py — the eval generation MATRIX: every checkpoint x {seen, unseen} sentences x 5 seeds,
denoised ggs_l1_0112 prompt. Names files c<ckpt>__<sent>__s<seed>.wav for scripts/eval_metrics.py.
Resumable (skips files already written). Run on the pod after pod_infer.sh + LoRA uploads.
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
OUT = "/workspace/out_eval"
CKPTS = {"150": "/workspace/lora150", "250": "/workspace/lora250",
         "350": "/workspace/lora350", "500": "/workspace/lora500"}
SENTS = {
    "nm":  "Namabhasa means offences are not completely gone. If offences will completely go, then pure name will rise",  # SEEN (training clip)
    "nv1": "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death.",                       # UNSEEN
    "nv2": "The pure devotee always remembers Krishna and never forgets Him for a single moment.",                         # UNSEEN
}
SEEDS = [42, 123, 7, 2024, 99]
os.makedirs(OUT, exist_ok=True)


def run(ck, lora_dir):
    info = json.load(open(f"{lora_dir}/lora_config.json"))
    cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
    print(f"=== loading checkpoint {ck} ===", file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                               lora_config=cfg, lora_weights_path=lora_dir)
    sr = m.tts_model.sample_rate
    for sid, text in SENTS.items():
        for s in SEEDS:
            out = f"{OUT}/c{ck}__{sid}__s{s}.wav"
            if os.path.exists(out):
                continue
            wav = m.generate(text=text, prompt_wav_path=REF, prompt_text=REF_TEXT, denoise=True, seed=s)
            sf.write(out, wav, sr)
            print(f"wrote {out}", file=sys.stderr)
    del m; gc.collect(); torch.cuda.empty_cache()


for ck, d in CKPTS.items():
    if os.path.isdir(d):
        run(ck, d)
    else:
        print(f"skip {ck} (no lora at {d})", file=sys.stderr)
print("ALL DONE")
