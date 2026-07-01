"""
score_refsweep.py — reference-selection experiment scorer. For each (reference config x model),
mean UNSEEN ECAPA cosine to his NOISY centroid (apples-to-apples with the 0.743 orig-data baseline).
Reads data/out/lecture1/refsweep/ (pulled from the pod). Run: .venv/Scripts/python.exe scripts/score_refsweep.py

Answers 4 questions the sweep was built for:
  (1) does a better reference beat the current 0112 baseline?              -> rows vs c0112n
  (2) calm-modal (0086) vs most-typical-but-shouty (0081)?                 -> c0086d vs c0081d
  (3) does the reference beat the LoRA?  (blind-eval hunch)                -> base vs lora150 columns
  (4) does denoising the REFERENCE help/hurt identity?                     -> c0112n vs c0112d
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
SWEEP = ROOT / "data/out/lecture1/refsweep"
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


# his noisy centroid — same selection as every clone-scorer, with ALL reference clips excluded
rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCL = {"ggs_l1_0112", "ggs_l1_0114", "ggs_l1_0086", "ggs_l1_0081"}
sel = [Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
ref = F.normalize(torch.stack([embed(CLIPS / n) for n in sel[:20]]).mean(0), dim=0)
ceil = float(np.mean([cos(embed(CLIPS / n), ref) for n in sel[20:30]]))

pat = re.compile(r"r(?P<rid>\w+?)__(?P<model>\w+?)__(?P<sent>\w+?)__s(?P<seed>\d+)\.wav")
agg = defaultdict(list)   # (rid, model) -> [unseen ecapa]
for p in sorted(SWEEP.glob("*.wav")):
    m = pat.match(p.name)
    if not m:
        continue
    agg[(m["rid"], m["model"])].append(cos(embed(p), ref))

REF_LABEL = {"c0112n": "0112 noisy+dn (BASELINE)", "c0112d": "0112 denoised",
             "c0086d": "0086 calm-modal", "c0081d": "0081 typical/shout"}
RIDS = ["c0112n", "c0112d", "c0086d", "c0081d"]
MODELS = ["base", "lora150"]


def fmt(v):
    return f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "—"


def mean_of(rid, mdl):
    v = agg[(rid, mdl)]
    return float(np.mean(v)) if v else None


print(f"\nHIS-OWN ECAPA ceiling: {ceil:.3f}   |   baseline to beat: 0.743 (orig-data LoRA-150, unseen)\n")
print(f"{'reference':28}{'base':>16}{'lora150':>16}")
for rid in RIDS:
    print(f"{REF_LABEL[rid]:28}{fmt(agg[(rid,'base')]):>16}{fmt(agg[(rid,'lora150')]):>16}")

print("\n=== READS ===")
best = max(((np.mean(v), rid, mdl) for (rid, mdl), v in agg.items() if v), default=(None, None, None))
if best[0] is not None:
    print(f"  best cell: {REF_LABEL[best[1]]} + {best[2]}  ->  {best[0]:.3f}   (vs 0.743 baseline: {best[0]-0.743:+.3f})")
for mdl in MODELS:
    a, b = mean_of("c0112n", mdl), mean_of("c0112d", mdl)
    if a and b:
        print(f"  reference-denoise ({mdl:7}): 0112 noisy {a:.3f} -> denoised {b:.3f}  ({b-a:+.3f})")
    c, d = mean_of("c0086d", mdl), mean_of("c0081d", mdl)
    if c and d:
        print(f"  calm vs shout   ({mdl:7}): 0086-calm {c:.3f}  vs  0081-shout {d:.3f}  ({c-d:+.3f})")
# does the reference beat the LoRA? best-ref base vs best-ref lora
for rid in RIDS:
    a, b = mean_of(rid, "base"), mean_of(rid, "lora150")
    if a and b:
        print(f"  reference vs LoRA @ {REF_LABEL[rid]:24}: base {a:.3f}  vs  lora150 {b:.3f}  ({b-a:+.3f})")
