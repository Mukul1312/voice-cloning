"""
score_denoise_retrain.py — did denoising the TRAINING audio move unseen-ECAPA off the ~0.74 plateau?

Scores the ORIGINAL-data and DENOISED-data eval matrices (same prompt/sentences/seeds, only the
training data differs) against his speaker centroid, two ways:
  - vs the NOISY centroid (his original clips)  -> apples-to-apples with the old run's metric
  - vs the DENOISED centroid (clips_dn)          -> the fair view for a clean-trained clone (confound fix)
Ranks by UNSEEN content only (overfitting-clean). Run: .venv/Scripts/python.exe scripts/score_denoise_retrain.py
"""
import json
import re
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data/out/lecture1/final"
CLIPS = FINAL / "clips"
CLIPS_DN = FINAL / "clips_dn"
EVAL_ORIG = ROOT / "data/out/lecture1/eval"
EVAL_DN = ROOT / "data/out/lecture1/eval_dn"
DEV = "cpu"

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(ROOT / ".venv/spkrec"), run_opts={"device": DEV},
                                     local_strategy=LocalStrategy.COPY)


def load16k(p):
    y, sr = sf.read(str(p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        y = resample_poly(y, 16000 // g, sr // g).astype("float32")
    return y


@torch.no_grad()
def embed(p):
    w = torch.from_numpy(np.ascontiguousarray(load16k(p))).unsqueeze(0)
    return F.normalize(clf.encode_batch(w.to(DEV)).squeeze(0).squeeze(0), dim=0)


def cos(a, b):
    return float(torch.dot(a, b))


# --- centroid clip selection (matches eval_metrics: English-dense, 5-12s, exclude prompt clips) ---
rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCL = {"ggs_l1_0112", "ggs_l1_0114"}
sel = [Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
ref_names, held_names = sel[:20], sel[20:30]


def centroid(base, names):
    return F.normalize(torch.stack([embed(base / n) for n in names]).mean(0), dim=0)


ref_noisy = centroid(CLIPS, ref_names)
ref_dn = centroid(CLIPS_DN, ref_names)
ceil_noisy = float(np.mean([cos(embed(CLIPS / n), ref_noisy) for n in held_names]))
ceil_dn = float(np.mean([cos(embed(CLIPS_DN / n), ref_dn) for n in held_names]))

pat = re.compile(r"c(?P<ckpt>\w+?)__(?P<sent>\w+?)__s(?P<seed>\d+)\.wav")


def score_dir(d, ref):
    agg = defaultdict(lambda: defaultdict(list))
    if not d.exists():
        return agg
    for p in sorted(d.glob("*.wav")):
        mm = pat.match(p.name)
        if not mm:
            continue
        cond = "SEEN" if mm["sent"] == "nm" else "UNSEEN"
        agg[mm["ckpt"]][cond].append(cos(embed(p), ref))
    return agg


orig_noisy = score_dir(EVAL_ORIG, ref_noisy)
dn_noisy = score_dir(EVAL_DN, ref_noisy)
dn_dn = score_dir(EVAL_DN, ref_dn)


def ms(v):
    return (np.mean(v), np.std(v), len(v)) if v else (None, None, 0)


def fmt(v):
    mu, sd, n = ms(v)
    return f"{mu:.3f}±{sd:.3f}" if mu is not None else "—"


def best_unseen(agg):
    cand = [(np.mean(agg[ck]["UNSEEN"]), ck) for ck in agg if agg[ck]["UNSEEN"]]
    return max(cand) if cand else (None, None)


ckpts = sorted(set(list(orig_noisy) + list(dn_noisy)), key=lambda x: int(x) if x.isdigit() else 999)
print(f"\nHIS-OWN ECAPA ceiling — noisy centroid {ceil_noisy:.3f} | denoised centroid {ceil_dn:.3f}")
print("\n=== UNSEEN content (the overfitting-clean number; higher = more him) ===")
print(f"{'ckpt':<6}{'ORIG-data vs noisy':<22}{'DN-data vs noisy':<22}{'DN-data vs DN-centroid':<24}")
for ck in ckpts:
    print(f"{ck:<6}{fmt(orig_noisy[ck]['UNSEEN']):<22}{fmt(dn_noisy[ck]['UNSEEN']):<22}{fmt(dn_dn[ck]['UNSEEN']):<24}")

print("\n=== SEEN content (Namabhasa — overfitting-confounded, reference only) ===")
print(f"{'ckpt':<6}{'ORIG-data vs noisy':<22}{'DN-data vs noisy':<22}{'DN-data vs DN-centroid':<24}")
for ck in ckpts:
    print(f"{ck:<6}{fmt(orig_noisy[ck]['SEEN']):<22}{fmt(dn_noisy[ck]['SEEN']):<22}{fmt(dn_dn[ck]['SEEN']):<24}")

bo = best_unseen(orig_noisy); bn = best_unseen(dn_noisy); bd = best_unseen(dn_dn)
print("\n=== VERDICT (best UNSEEN per run) ===")
print(f"  ORIGINAL data, vs noisy centroid:   {bo[0]:.3f}  (step {bo[1]})" if bo[0] else "  ORIGINAL: —")
print(f"  DENOISED data, vs noisy centroid:   {bn[0]:.3f}  (step {bn[1]})   [apples-to-apples]" if bn[0] else "  DENOISED noisy: —")
print(f"  DENOISED data, vs DENOISED centroid:{bd[0]:.3f}  (step {bd[1]})   [confound-corrected]" if bd[0] else "  DENOISED dn: —")
if bo[0] and bn[0]:
    print(f"\n  apples-to-apples delta (DN - ORIG, vs noisy centroid): {bn[0] - bo[0]:+.3f}")
