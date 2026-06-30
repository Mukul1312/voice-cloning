"""
ecapa_score.py — the rigorous clone-identity metric we always planned (finetune-plan.md Phase 3):
ECAPA-TDNN speaker-embedding COSINE similarity, clone vs his real voice. Reuses the exact embedder
from cloud/pod_dataprep.py. ECAPA needs 16 kHz mono, so every clip routes through load_wav (VoxCPM
clones are 48 kHz -> downsampled). Higher cosine = more "him".
"""
import json
import re
import sys
from pathlib import Path

from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data/out/lecture1/final/clips"
SAMP = ROOT / "data/out/lecture1/samples"
CMP = ROOT / "data/out/lecture1/cmp"
DEV = "cpu"

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(ROOT / ".venv/spkrec"), run_opts={"device": DEV},
                                     local_strategy=LocalStrategy.COPY)  # Windows: no symlinks


def load_wav(path, sr_t=16000):
    y, sr = sf.read(str(path), dtype="float32")          # (T,) or (T, C)
    if y.ndim > 1:
        y = y.mean(axis=1)                                # downmix to mono
    if sr != sr_t:
        g = gcd(sr, sr_t)
        y = resample_poly(y, sr_t // g, sr // g).astype("float32")  # 48 kHz clone -> 16 kHz
    return torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)  # (1, T) for encode_batch


@torch.no_grad()
def embed(path):
    e = clf.encode_batch(load_wav(path).to(DEV)).squeeze(0).squeeze(0)
    return F.normalize(e, dim=0)


def cosine(a, b):
    return float(torch.dot(a, b))


def centroid(paths):
    return F.normalize(torch.stack([embed(p) for p in paths]).mean(0), dim=0)


# --- his real-voice centroid (English-dense clips, EXCLUDING the prompt clips) ---
rows = [json.loads(l) for l in (ROOT / "data/out/lecture1/final/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCLUDE = {"ggs_l1_0112", "ggs_l1_0114"}   # used as voice prompts -> keep out of the reference
cand = [CLIPS / Path(r["audio"]).name for r in rows
        if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCLUDE]
ref = centroid(cand[:20])
held = cand[20:30]                          # held-out REAL clips -> the realistic cosine "ceiling"
ceiling = [cosine(embed(w), ref) for w in held]
print(f"\nreference: centroid of {min(20,len(cand))} real clips (prompt clips excluded)")
print(f"HIS-OWN ceiling (held-out real vs centroid): mean {sum(ceiling)/len(ceiling):.3f}  "
      f"min {min(ceiling):.3f}  max {max(ceiling):.3f}   <- a clone can't beat this")

CLONES = [
    ("250  Namabhasa  SEEN", CMP / "s250_nm_s42.wav"),
    ("350  Namabhasa  SEEN", CMP / "s350_nm_s42.wav"),
    ("250  novel  UNSEEN  s42", CMP / "s250_nv_s42.wav"),
    ("250  novel  UNSEEN  s123", CMP / "s250_nv_s123.wav"),
    ("350  novel  UNSEEN  s42", CMP / "s350_nv_s42.wav"),
    ("350  novel  UNSEEN  s123", CMP / "s350_nv_s123.wav"),
    ("250  denoised-prompt  T1", SAMP / "s250_dnp_T1.wav"),
    ("250  denoised-prompt  T2", SAMP / "s250_dnp_T2.wav"),
    ("250  raw-prompt  T1", SAMP / "s250_np_T1.wav"),
    ("base  denoised-prompt", SAMP / "base_dnp_T1.wav"),
    ("150  plain (baked)", SAMP / "s150_plain1_with_lora.wav"),
    ("250  plain (baked)", SAMP / "s250_plain1_with_lora.wav"),
    ("350  plain (baked)", SAMP / "s350_plain1_with_lora.wav"),
    ("500  plain (baked)", SAMP / "s500_plain1_with_lora.wav"),
    ("base plain (CONTROL)", SAMP / "s150_plain1_lora_disabled.wav"),
]
print("\n=== ECAPA cosine to his real voice  (higher = more him) ===")
scored = []
for label, p in CLONES:
    if not p.exists():
        print(f"  (missing {p.name})"); continue
    scored.append((cosine(embed(p), ref), label))
scored.sort(reverse=True)
for c, label in scored:
    print(f"  {c:.3f}   {label}")

# parallel same-words: clone Namabhasa vs his REAL Namabhasa (ggs_l1_0112)
real_nm = embed(CLIPS / "ggs_l1_0112.wav")
print("\n=== parallel SAME-WORDS: cos(clone Namabhasa, his REAL Namabhasa) ===")
for label, p in [("250 Namabhasa", CMP / "s250_nm_s42.wav"), ("350 Namabhasa", CMP / "s350_nm_s42.wav")]:
    print(f"  {cosine(embed(p), real_nm):.3f}   {label}")
