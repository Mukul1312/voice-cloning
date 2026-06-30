"""
denoise_postproc.py — turn Resemble-Enhance output back into VoxCPM2 training clips.

Resemble-Enhance emits 44.1 kHz; VoxCPM2 trains at 16 kHz. This does ONLY linear DSP
(no generative step): resample 44.1k -> 16k, peak-normalize to 0.95 (the prep default we
saw in Mixomo/VoxCPM2_Simple_GUI), and trim trailing silence to <=0.5 s (the "infinite
generation" guard noted in that repo). Then it rewrites the manifest to point at the
denoised clips, preserving every original field except audio (path) + duration (recomputed).

Identity-safety: pair this with `resemble-enhance --denoise_only` (denoiser, NOT the
generative enhancer) so his timbre is preserved. This script never alters the voice.

Usage (run on the pod after pod_denoise.sh step 2):
  python cloud/denoise_postproc.py \
      --src_manifest data/out/lecture1/final/train.voxcpm.jsonl \
      --denoised_dir data/out/lecture1/final/clips_dn_44k \
      --out_clips    data/out/lecture1/final/clips_dn \
      --out_manifest data/out/lecture1/final/train_dn.voxcpm.jsonl
"""
import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
PEAK = 0.95           # peak-normalization target (community prep default)
MAX_TRAIL_SIL = 0.5   # keep at most this many seconds of trailing silence
SIL_DB = -40.0        # |sample| below peak*10^(SIL_DB/20) counts as silence


def load_mono(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def to_16k(y, sr):
    if sr == TARGET_SR:
        return y
    g = gcd(sr, TARGET_SR)
    return resample_poly(y, TARGET_SR // g, sr // g).astype("float32")


def trim_trailing_silence(y, sr):
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y))) or 1.0
    thr = peak * (10.0 ** (SIL_DB / 20.0))
    above = np.where(np.abs(y) > thr)[0]
    if len(above) == 0:
        return y
    keep = min(len(y), int(above[-1]) + 1 + int(MAX_TRAIL_SIL * sr))
    return y[:keep]


def peak_norm(y):
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return (y / m * PEAK).astype("float32") if m > 1e-6 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_manifest", required=True, help="original manifest (source of text)")
    ap.add_argument("--denoised_dir", required=True, help="Resemble-Enhance 44.1k output dir")
    ap.add_argument("--out_clips", required=True, help="dir for 16k denoised clips")
    ap.add_argument("--out_manifest", required=True, help="new manifest -> denoised clips")
    a = ap.parse_args()

    den = Path(a.denoised_dir)
    outc = Path(a.out_clips)
    outc.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.src_manifest, encoding="utf-8") if l.strip()]
    written, missing = 0, []
    with open(a.out_manifest, "w", encoding="utf-8") as mf:
        for r in rows:
            name = Path(r["audio"]).name
            src = den / name
            if not src.exists():
                missing.append(name)
                continue
            y, sr = load_mono(src)
            y = to_16k(y, sr)
            y = trim_trailing_silence(y, TARGET_SR)
            y = peak_norm(y)
            dst = (outc / name).resolve()
            sf.write(str(dst), y, TARGET_SR)
            nr = dict(r)                       # preserve text + any extra fields
            nr["audio"] = str(dst)             # absolute -> CWD-independent on the pod
            nr["duration"] = round(len(y) / TARGET_SR, 2)
            mf.write(json.dumps(nr, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} denoised 16k clips -> {outc}")
    print(f"manifest -> {a.out_manifest}")
    if missing:
        print(f"WARNING: {len(missing)} clips missing from denoised_dir (first 5): {missing[:5]}")


if __name__ == "__main__":
    main()
