"""
generate.py — PRODUCTION: speak any English sentence in Gour Govinda Swami's voice.

Winning recipe (locked 2026-07-01 after the reference sweep): LoRA-150 + the DENOISED ggs_l1_0081
reference clip. Unseen-content ECAPA 0.806 (his own ceiling 0.825; +0.063 over the old baseline),
and ear-confirmed the best of the five references — seed 42 is especially good.

  python cloud/generate.py --text "Real intelligence is to inquire about Krishna."
  python cloud/generate.py --text "..." --seeds 42 123 7   # a few takes to pick the best by ear
  python cloud/generate.py --text "..." --seed 42 --out /workspace/hello

Run on the pod after: pod_infer.sh (env + VoxCPM2 base) + upload the LoRA to /workspace/lora150.
"""
import argparse
import json
import os
import sys

import soundfile as sf
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

BASE = "/workspace/VoxCPM2"
LORA = "/workspace/lora150"                                   # upload cloud/lora_ckpts/step_0000150 here
REF = "/workspace/voice-cloning/data/out/lecture1/final/clips_dn/ggs_l1_0081.wav"   # the winning reference (denoised)
REF_TEXT = ('thats the process - and an intelligent disciple like Arjuna says, "No Guru Maharaja '
            'whatever you say Ill do. I have no other thought." This is the proper use of independence,')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="the English sentence to speak in his voice")
    ap.add_argument("--seed", type=int, default=42, help="best default from the ear-check")
    ap.add_argument("--seeds", type=int, nargs="+", help="generate several takes (overrides --seed)")
    ap.add_argument("--out", default="/workspace/ggs_out", help="output path prefix (no .wav)")
    a = ap.parse_args()

    info = json.load(open(f"{LORA}/lora_config.json"))
    cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
    print("loading VoxCPM2 + LoRA-150 ...", file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                               lora_config=cfg, lora_weights_path=LORA)
    sr = m.tts_model.sample_rate

    outdir = os.path.dirname(a.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    seeds = a.seeds or [a.seed]
    for s in seeds:
        wav = m.generate(text=a.text, prompt_wav_path=REF, prompt_text=REF_TEXT, denoise=False, seed=s)
        out = f"{a.out}_s{s}.wav" if len(seeds) > 1 else f"{a.out}.wav"
        sf.write(out, wav, sr)
        print(f"wrote {out}", file=sys.stderr)
    print("DONE")


if __name__ == "__main__":
    main()
