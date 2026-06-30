"""
infer_pretest.py — PRONUNCIATION pre-test for the diary Short.
Generates a handful of representative diary sentences (covering every tricky devotional/Sanskrit term
+ the spoken date) in his cloned voice, several seeds each, so we can hear whether the clone carries
this text with dignity BEFORE building the full production pipeline. Champion checkpoint (step 150).

Names: {label}__s{seed}.wav -> pulled + auditioned by scripts/build_pretest_gallery.py.
Run on the pod after pod_infer.sh + scp of the LoRA to /workspace/lora_champion.
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
LORA = "/workspace/lora_champion"
REF = "/workspace/voice-cloning/data/out/lecture1/final/clips/ggs_l1_0112.wav"
REF_TEXT = ("Namabhasa means offences are not completely gone. If offences will completely go, "
            "then pure name will rise, sun will arise, moon will rise, then prema will come")
OUT = "/workspace/out_pretest"

# diacritics stripped for the TTS; date written out so it's spoken, not read as digits.
SENTS = {
    "date":     "May eighth, nineteen seventy five, Thursday.",
    "vow":      "This servant took a vow not to eat anything at night except a little maha water with Tulasi.",
    "iskcon":   "To open an ISKCON centre in Odisha.",
    "prema":    "For receiving complete and uninterrupted Krishna prema.",
    "prasadam": "He will not accept any Prasadam at night.",
    "lila":     "But what an astonishing leela You are performing.",
    "sannyasi": "This servant did not desire to perform such type of severe austerity in front of a sannyasi present here.",
    "gopala":   "Oh my Lord, Gopala! Please shower Your mercy.",
    "gauranga": "May this servant achieve success in the service of Sri Sri Guru and Gauranga. Hare Krishna.",
}
SEEDS = [42, 123, 7, 2024, 99]
os.makedirs(OUT, exist_ok=True)

info = json.load(open(f"{LORA}/lora_config.json"))
cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
print("=== loading champion checkpoint ===", file=sys.stderr)
m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                           lora_config=cfg, lora_weights_path=LORA)
sr = m.tts_model.sample_rate
for sid, text in SENTS.items():
    for s in SEEDS:
        out = f"{OUT}/{sid}__s{s}.wav"
        if os.path.exists(out):
            continue
        wav = m.generate(text=text, prompt_wav_path=REF, prompt_text=REF_TEXT, denoise=True, seed=s)
        sf.write(out, wav, sr)
        print(f"wrote {out}", file=sys.stderr)
del m; gc.collect(); torch.cuda.empty_cache()
print("ALL DONE")
