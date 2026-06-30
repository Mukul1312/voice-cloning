"""
ecapa_denoise_check.py — IDENTITY SAFETY-GATE for denoising (run BEFORE the denoised retrain).

Denoising is only worth doing if it removes NOISE without altering HIM. This proves it:
build his ECAPA speaker centroid from the ORIGINAL clips, then score both the ORIGINAL and the
DENOISED version of the SAME held-out clips against that centroid. If the denoised mean-cosine
holds (>= original - 0.03), identity is preserved -> safe to retrain. A real drop means the
denoiser is reshaping his timbre -> stop, use gentler settings or a non-generative denoiser
(e.g. DeepFilterNet) instead.

Reuses the exact embedder from scripts/ecapa_score.py. Run:  python scripts/ecapa_denoise_check.py
"""
import sys
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
ORIG = ROOT / "data/out/lecture1/final/clips"
DN = ROOT / "data/out/lecture1/final/clips_dn"
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


def main():
    if not DN.exists():
        print(f"no denoised clips at {DN} — run cloud/pod_denoise.sh first")
        return
    names = sorted(p.name for p in ORIG.glob("*.wav"))
    if len(names) < 25:
        print(f"only {len(names)} original clips found at {ORIG}")
        return
    # centroid from his ORIGINAL voice (first 20 clips)
    ref = F.normalize(torch.stack([embed(ORIG / n) for n in names[:20]]).mean(0), dim=0)
    # compare ORIGINAL vs DENOISED on the SAME held-out clips (the rest that were denoised)
    held = [n for n in names[20:] if (DN / n).exists()]
    if not held:
        print("no overlapping denoised clips to compare")
        return
    co = [cos(embed(ORIG / n), ref) for n in held]
    cd = [cos(embed(DN / n), ref) for n in held]
    mo, md = float(np.mean(co)), float(np.mean(cd))
    print(f"\nheld-out clips compared: {len(held)}")
    print(f"ORIGINAL  mean cosine to his centroid: {mo:.3f}")
    print(f"DENOISED  mean cosine to his centroid: {md:.3f}")
    print(f"delta (denoised - original):           {md - mo:+.3f}")
    if md >= mo - 0.03:
        print("\nPASS — identity preserved. Safe to retrain on the denoised data.")
    else:
        print("\nWARN — denoiser shifted his voice (cosine dropped > 0.03).")
        print("       Use gentler denoising or a non-generative denoiser (DeepFilterNet) instead.")


if __name__ == "__main__":
    main()
