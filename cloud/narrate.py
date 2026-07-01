"""
narrate.py — generate a full narration in GGS's voice from a chunks file (one already-transliterated
utterance per line). Winning recipe (locked 2026-07-01): LoRA-150 + denoised ggs_l1_0081, seed 42.
Writes per-chunk wavs + a concatenated full.wav (with a short silence between chunks so it breathes).

  python cloud/narrate.py --chunks /workspace/voice-cloning/data/narration/diary_chunks.txt \
      --out /workspace/narration --seed 42
Run on the pod after pod_infer.sh + LoRA upload to /workspace/lora150.
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

BASE = "/workspace/VoxCPM2"
LORA = "/workspace/lora150"
REF = "/workspace/voice-cloning/data/out/lecture1/final/clips_dn/ggs_l1_0081.wav"
REF_TEXT = ('thats the process - and an intelligent disciple like Arjuna says, "No Guru Maharaja '
            'whatever you say Ill do. I have no other thought." This is the proper use of independence,')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="text file, one transliterated utterance per line")
    ap.add_argument("--out", default="/workspace/narration")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gap", type=float, default=0.45, help="silence seconds between chunks")
    a = ap.parse_args()

    lines = [l.strip() for l in open(a.chunks, encoding="utf-8") if l.strip() and not l.lstrip().startswith("#")]
    os.makedirs(a.out, exist_ok=True)
    info = json.load(open(f"{LORA}/lora_config.json"))
    cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
    print(f"loading VoxCPM2 + LoRA-150 ... ({len(lines)} chunks, seed {a.seed})", file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                               lora_config=cfg, lora_weights_path=LORA)
    sr = m.tts_model.sample_rate
    gap = np.zeros(int(a.gap * sr), dtype="float32")

    full = []
    for i, text in enumerate(lines, 1):
        wav = np.asarray(m.generate(text=text, prompt_wav_path=REF, prompt_text=REF_TEXT,
                                    denoise=False, seed=a.seed), dtype="float32")
        sf.write(f"{a.out}/chunk_{i:02d}.wav", wav, sr)
        full.append(wav)
        full.append(gap)
        print(f"[{i}/{len(lines)}] {text[:64]}", file=sys.stderr)
    sf.write(f"{a.out}/full.wav", np.concatenate(full), sr)
    print(f"DONE -> {a.out}/full.wav  (+ {len(lines)} chunk_*.wav)")


if __name__ == "__main__":
    main()
