"""
infer_dn.py — denoised-reference prompted generation from the step-250 LoRA (the cheap-squeeze
confirmation). Loads VoxCPM2 + step-250 LoRA + the zipenhancer denoiser, then generates target
lines conditioned on his real clip (ggs_l1_0112) with the prompt DENOISED.

Outputs -> /workspace/out_dn/  (pulled down + dropped into the scorecard)
Run on the pod:  source /workspace/venv/bin/activate && python /workspace/infer_dn.py
"""
import json
import os
import sys
import soundfile as sf
from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig

LORA = "/workspace/lora250"
BASE = "/workspace/VoxCPM2"
REF = "/workspace/voice-cloning/data/out/lecture1/final/clips/ggs_l1_0112.wav"
REF_TEXT = ("Namabhasa means offences are not completely gone. If offences will completely go, "
            "then pure name will rise, sun will arise, moon will rise, then prema will come")
OUT = "/workspace/out_dn"
T1 = "Real intelligence is to give up this illusion and inquire about Krishna."
T2 = "Chant the holy name with attention and devotion, and the Lord will reveal Himself."

assert os.path.exists(REF), f"reference clip missing: {REF}"
os.makedirs(OUT, exist_ok=True)
info = json.load(open(f"{LORA}/lora_config.json"))
cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None

print("loading VoxCPM2 + step-250 LoRA + zipenhancer denoiser...", file=sys.stderr)
model = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                               lora_config=cfg, lora_weights_path=LORA)
sr = model.tts_model.sample_rate
print(f"loaded; output sample_rate={sr}", file=sys.stderr)


def gen(text, denoise, out, seed=42):
    wav = model.generate(text=text, prompt_wav_path=REF, prompt_text=REF_TEXT,
                         denoise=denoise, seed=seed)
    sf.write(out, wav, sr)
    print(f"wrote {out}  ({len(wav)/sr:.2f}s, denoise={denoise})", file=sys.stderr)


# LoRA (step 250) + DENOISED prompt — the candidates to listen to
gen(T1, True, f"{OUT}/s250_dnp_T1.wav")
gen(T2, True, f"{OUT}/s250_dnp_T2.wav")
# A/B control: same checkpoint + RAW (noisy) prompt, same seed — isolates the denoiser's effect
gen(T1, False, f"{OUT}/s250_np_T1.wav")
# base model + denoised prompt (LoRA disabled) — does the LoRA still add anything in the prompted regime?
model.set_lora_enabled(False)
gen(T1, True, f"{OUT}/base_dnp_T1.wav")
print("ALL DONE")
