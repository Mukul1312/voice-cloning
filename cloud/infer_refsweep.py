"""
infer_refsweep.py — reference-selection experiment (Option B). Tests whether the PROMPT/reference
clip is the identity lever behind the ~0.74 plateau (our blind eval hinted base+prompt >= LoRA+prompt).
For {base, LoRA-150} x {4 reference configs} x {2 UNSEEN sentences} x {3 seeds} -> 48 clips.
Names r{refid}__{model}__{sent}__s{seed}.wav  -> scored locally by scripts/score_refsweep.py.

Reference configs (from the ear-check + the representativeness analysis, cos = ECAPA-to-centroid):
  c0112n = ggs_l1_0112 NOISY    + denoise=True   -> EXACT current baseline anchor (cos 0.795, atypical/high-pitch)
  c0112d = ggs_l1_0112 DENOISED + denoise=False  -> same clip cleaned (ISOLATES the reference-denoise effect)
  c0086d = ggs_l1_0086 DENOISED + denoise=False  -> calm modal register (F0~133, ear-verified: no shouting), cos 0.875
  c0081d = ggs_l1_0081 DENOISED + denoise=False  -> most him-typical (cos 0.939) BUT has a shout (probes register bleed)
Model = the ORIGINAL noisy-trained LoRA at step 150 (best unseen, 0.743). NOTE: LoRA checkpoints are
gitignored, so UPLOAD cloud/lora_ckpts/step_0000150 -> /workspace/lora150 on the pod first (same as
infer_eval.py's convention). Resumable (skips files already written). Run after repo clone + VoxCPM2 download.
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
REPO = "/workspace/voice-cloning"
CLIPS = f"{REPO}/data/out/lecture1/final/clips"
CLIPS_DN = f"{REPO}/data/out/lecture1/final/clips_dn"
LORA150 = "/workspace/lora150"   # UPLOAD cloud/lora_ckpts/step_0000150 here (gitignored; infer_eval.py convention)
OUT = "/workspace/out_refsweep"

T0112 = ("Namabhasa means offences are not completely gone. If offences will completely go, "
         "then pure name will rise, sun will arise, moon will rise, then prema will come")
T0086 = ("guru says that karishye, I have to do - may be very unfavorable, is very unfavorable "
         "on the part of Arjuna to kill his kith and kin but he did it - this is love. "
         "This is true love, pure love.")
T0081 = ('thats the process - and an intelligent disciple like Arjuna says, "No Guru Maharaja '
         'whatever you say Ill do. I have no other thought." This is the proper use of independence,')

# (refid, wav_path, ref_text, denoise_flag)
REFS = [
    ("c0112n", f"{CLIPS}/ggs_l1_0112.wav",    T0112, True),
    ("c0112d", f"{CLIPS_DN}/ggs_l1_0112.wav", T0112, False),
    ("c0086d", f"{CLIPS_DN}/ggs_l1_0086.wav", T0086, False),
    ("c0081d", f"{CLIPS_DN}/ggs_l1_0081.wav", T0081, False),
    ("c0081n", f"{CLIPS}/ggs_l1_0081.wav",    T0081, True),   # follow-up: NOISY 0081 (Q4: noisy > denoised on same clip)
]
SENTS = {  # UNSEEN (same as every prior eval, for apples-to-apples)
    "nv1": "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death.",
    "nv2": "The pure devotee always remembers Krishna and never forgets Him for a single moment.",
}
SEEDS = [42, 123, 7]
MODELS = ["base", "lora150"]
os.makedirs(OUT, exist_ok=True)


def load_model(which):
    if which == "base":
        return VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False)
    info = json.load(open(f"{LORA150}/lora_config.json"))
    cfg = LoRAConfig(**info["lora_config"]) if info.get("lora_config") else None
    return VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False,
                                  lora_config=cfg, lora_weights_path=LORA150)


for which in MODELS:
    pending = [(rid, wav, rtext, dn, sid, s)
               for (rid, wav, rtext, dn) in REFS for sid in SENTS for s in SEEDS
               if not os.path.exists(f"{OUT}/r{rid}__{which}__{sid}__s{s}.wav")]
    if not pending:
        print(f"skip {which} (all done)", file=sys.stderr)
        continue
    print(f"=== loading {which} ===", file=sys.stderr)
    m = load_model(which)
    sr = m.tts_model.sample_rate
    for (rid, wav, rtext, dn, sid, s) in pending:
        w = m.generate(text=SENTS[sid], prompt_wav_path=wav, prompt_text=rtext, denoise=dn, seed=s)
        sf.write(f"{OUT}/r{rid}__{which}__{sid}__s{s}.wav", w, sr)
        print(f"wrote r{rid}__{which}__{sid}__s{s}.wav", file=sys.stderr)
    del m
    gc.collect()
    torch.cuda.empty_cache()
print("ALL DONE")
