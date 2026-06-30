"""
score_curve.py — read the learning-curve eval sweep and answer: does best UNSEEN-ECAPA rise with
training-set size? Builds his noisy centroid, scores every n{N}_k{step}__{sent}__s{seed}.wav, takes
each subset's BEST checkpoint (over the swept steps) on unseen content, and prints the curve + a slope
read. Rising at n140 -> data-starved, quantity is a real lever. Flat -> saturated, it isn't.

Reads data/out/lecture1/curve/ (pulled from the pod). Run: .venv/Scripts/python.exe scripts/score_curve.py
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
CURVE = ROOT / "data/out/lecture1/curve"
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


# his noisy centroid (same selection as the other scorers)
rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCL = {"ggs_l1_0112", "ggs_l1_0114"}
sel = [Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
ref = F.normalize(torch.stack([embed(CLIPS / n) for n in sel[:20]]).mean(0), dim=0)
ceil = float(np.mean([cos(embed(CLIPS / n), ref) for n in sel[20:30]]))

pat = re.compile(r"n(?P<N>\d+)_k(?P<step>\d+)__(?P<sent>\w+?)__s(?P<seed>\d+)\.wav")
agg = defaultdict(list)   # (N, step) -> [unseen ecapa]
for p in sorted(CURVE.glob("*.wav")):
    m = pat.match(p.name)
    if not m:
        continue
    agg[(int(m["N"]), int(m["step"]))].append(cos(embed(p), ref))

byN = defaultdict(dict)
for (N, step), vals in agg.items():
    byN[N][step] = (float(np.mean(vals)), float(np.std(vals)), len(vals))

print(f"\nHIS-OWN ECAPA ceiling: {ceil:.3f}")
print("\n=== per (subset, checkpoint) — UNSEEN ECAPA mean±std ===")
for N in sorted(byN):
    row = "  ".join(f"k{st}:{byN[N][st][0]:.3f}" for st in sorted(byN[N]))
    print(f"  n{N:<4} {row}")

print("\n=== LEARNING CURVE — best UNSEEN ECAPA per subset (its own peak checkpoint) ===")
curve = []
for N in sorted(byN):
    best_step = max(byN[N], key=lambda s: byN[N][s][0])
    mu, sd, n = byN[N][best_step]
    curve.append((N, mu, best_step))
    bar = "█" * int(round((mu - 0.5) / (ceil - 0.5) * 40)) if ceil > 0.5 else ""
    print(f"  n{N:<4} {mu:.3f} ±{sd:.3f}  (best @ step {best_step})  {bar}")

if len(curve) >= 2:
    print("\n=== SLOPE READ ===")
    print(f"  n{curve[0][0]} -> n{curve[-1][0]}:  {curve[0][1]:.3f} -> {curve[-1][1]:.3f}  (delta {curve[-1][1]-curve[0][1]:+.3f})")
    tail = curve[-1][1] - curve[-2][1]
    print(f"  last step ({curve[-2][0]} -> {curve[-1][0]}): {tail:+.3f}  -> "
          + ("STILL RISING: more data should help (quantity is a real lever)." if tail > 0.01
             else "FLAT/SATURATED: more of the SAME data won't help (rethink the lever)."))
    print(f"  (his ceiling {ceil:.3f}; gap from best subset to ceiling: {ceil - curve[-1][1]:+.3f})")
