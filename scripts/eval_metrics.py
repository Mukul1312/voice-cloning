"""
eval_metrics.py — the rigorous clone-eval engine (the metrics we always planned, finally wired in).

Per the audit:
  - PRIMARY (identity/timbre): ECAPA-TDNN speaker-embedding COSINE to his real-voice centroid.
    Content-independent -> the clean cross-checkpoint / SEEN-vs-UNSEEN ruler. ECAPA needs 16 kHz mono.
  - PARALLEL only (same words, clone vs his REAL clip): MCD (DTW-aligned mel-cepstral distortion, timbre)
    and F0-RMSE in semitones (cadence). Only valid where we have his real audio of the same line (Namabhasa).
  - CADENCE everywhere: clone F0 median vs his (a distribution read for content with no parallel real).
  - Multi-seed: aggregate mean +/- std; pick checkpoints on UNSEEN content only (overfitting guard).

Reads clips from data/out/lecture1/eval/  named  c<ckpt>__<sent>__s<seed>.wav
(bootstrapped below with the clips we already generated). Run:  .venv/Scripts/python.exe scripts/eval_metrics.py
"""
import json
import re
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import types
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
# librosa lazy-loads its submodules; resolve them NOW while sys.modules is clean. Importing
# speechbrain (below) registers a lazy `k2` stub, and librosa's lazy_loader later walks frames
# via inspect.stack and trips on that missing module -> ImportError. Force-resolve + stub k2.
import librosa, librosa.feature, librosa.sequence, librosa.core.audio  # noqa
_ = (librosa.feature.mfcc, librosa.sequence.dtw)
import parselmouth
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
_k2 = types.ModuleType("k2"); _k2.__file__ = "<stub>"; sys.modules["k2"] = _k2  # defensive

import importlib.util
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
CLIPS = ROOT / "data/out/lecture1/final/clips"
EVAL = ROOT / "data/out/lecture1/eval"
DEV = "cpu"

# reuse the SHS-based robust F0 from the anatomy builder
_spec = importlib.util.spec_from_file_location("bva", ROOT / "scripts" / "build_voice_anatomy.py")
bva = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bva)

# --- sentence registry: which are SEEN (in training) and which have a real parallel reference ---
REAL_NM = CLIPS / "ggs_l1_0112.wav"
SENTS = {
    "nm":  {"seen": True,  "real": REAL_NM},   # Namabhasa = a training clip
    "nv":  {"seen": False, "real": None},
    "nv1": {"seen": False, "real": None},
    "nv2": {"seen": False, "real": None},
}

# ---------------- ECAPA ----------------
clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(ROOT / ".venv/spkrec"), run_opts={"device": DEV},
                                     local_strategy=LocalStrategy.COPY)


def load16k(path):
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        y = resample_poly(y, 16000 // g, sr // g).astype("float32")
    return y


@torch.no_grad()
def embed(path):
    wav = torch.from_numpy(np.ascontiguousarray(load16k(path))).unsqueeze(0)
    e = clf.encode_batch(wav.to(DEV)).squeeze(0).squeeze(0)
    return F.normalize(e, dim=0)


def cosine(a, b):
    return float(torch.dot(a, b))


def centroid(paths):
    return F.normalize(torch.stack([embed(p) for p in paths]).mean(0), dim=0)


# ---------------- parallel timbre + cadence ----------------
def mcd(clone_path, real_path):
    """DTW-aligned mel-cepstral distortion (dB). Lower = closer timbre. Same-words only."""
    yc, yr = load16k(clone_path), load16k(real_path)
    mc = librosa.feature.mfcc(y=yc, sr=16000, n_mfcc=25)[1:]   # drop energy coeff 0
    mr = librosa.feature.mfcc(y=yr, sr=16000, n_mfcc=25)[1:]
    _, wp = librosa.sequence.dtw(X=mc, Y=mr, metric="euclidean")
    diffs = [mc[:, i] - mr[:, j] for i, j in wp]
    d = np.mean([np.sqrt(np.sum(df * df)) for df in diffs])
    return float((10.0 / np.log(10)) * np.sqrt(2) * d)


def _f0_hz(path):
    snd = parselmouth.Sound(str(path))
    n = int(snd.get_total_duration() / bva.DT)
    arr = bva.robust_f0(snd, n)                 # nan = unvoiced
    return arr[~np.isnan(arr)]                   # voiced Hz only


def f0_rmse_semitones(clone_path, real_path):
    """DTW-aligned F0 RMSE in semitones (base-independent). Lower = closer cadence. Same-words only."""
    vc, vr = _f0_hz(clone_path), _f0_hz(real_path)
    if len(vc) < 5 or len(vr) < 5:
        return None
    _, wp = librosa.sequence.dtw(X=vc[None, :], Y=vr[None, :], metric="euclidean")
    st = np.array([12.0 * np.log2(vc[i] / vr[j]) for i, j in wp])
    return float(np.sqrt(np.mean(st * st)))


def f0_median(path):
    return float(np.median(_f0_hz(path))) if len(_f0_hz(path)) else None


# ---------------- bootstrap current clips into the eval dir ----------------
def bootstrap_current():
    """Copy the clips we already generated into EVAL with the c<ckpt>__<sent>__s<seed> convention."""
    import shutil
    EVAL.mkdir(parents=True, exist_ok=True)
    CMP = ROOT / "data/out/lecture1/cmp"
    mapping = {
        "s250_nm_s42.wav": "c250__nm__s42.wav", "s350_nm_s42.wav": "c350__nm__s42.wav",
        "s250_nv_s42.wav": "c250__nv__s42.wav", "s250_nv_s123.wav": "c250__nv__s123.wav",
        "s350_nv_s42.wav": "c350__nv__s42.wav", "s350_nv_s123.wav": "c350__nv__s123.wav",
    }
    for src, dst in mapping.items():
        if (CMP / src).exists() and not (EVAL / dst).exists():
            shutil.copyfile(CMP / src, EVAL / dst)


def main():
    # bootstrap_current()  # superseded by the full 60-clip matrix already pulled into EVAL
    # his centroid (English-dense clips, prompt clips excluded), + his own ceiling
    rows = [json.loads(l) for l in (ROOT / "data/out/lecture1/final/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split()); tok = re.compile(r"[a-z']+")
    ef = lambda t: (sum(1 for x in tok.findall(t.lower()) if x in eng) / len(tok.findall(t.lower()))) if tok.findall(t.lower()) else 0
    EXCL = {"ggs_l1_0112", "ggs_l1_0114"}
    cand = [CLIPS / Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
    ref = centroid(cand[:20])
    ceil = [cosine(embed(w), ref) for w in cand[20:30]]
    print(f"\nHIS-OWN ECAPA ceiling (held-out real vs centroid): mean {np.mean(ceil):.3f}  range {min(ceil):.3f}-{max(ceil):.3f}")

    pat = re.compile(r"c(?P<ckpt>\w+?)__(?P<sent>\w+?)__s(?P<seed>\d+)\.wav")
    perclip = []
    for p in sorted(EVAL.glob("*.wav")):
        m = pat.match(p.name)
        if not m:
            continue
        ck, sent, seed = m["ckpt"], m["sent"], int(m["seed"])
        meta = SENTS.get(sent, {"seen": False, "real": None})
        rec = dict(ckpt=ck, sent=sent, seed=seed, seen=meta["seen"],
                   ecapa=cosine(embed(p), ref), f0=f0_median(p))
        if meta["real"]:
            rec["mcd"] = mcd(p, meta["real"]); rec["f0rmse"] = f0_rmse_semitones(p, meta["real"])
        perclip.append(rec)

    # aggregate by (ckpt, seen/unseen)
    agg = defaultdict(lambda: defaultdict(list))
    for r in perclip:
        cond = "SEEN" if r["seen"] else "UNSEEN"
        agg[r["ckpt"]][cond].append(r)

    def ms(vals):
        vals = [v for v in vals if v is not None]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (None, None, 0)

    print("\n=== per-checkpoint, by condition (ECAPA cosine to his voice; higher=closer) ===")
    print(f"{'ckpt':<6}{'cond':<8}{'ECAPA mean±std (n)':<26}{'F0 med':<9}{'MCD':<8}{'F0-RMSE st':<10}")
    rank = {}
    for ck in sorted(agg):
        for cond in ("UNSEEN", "SEEN"):
            recs = agg[ck][cond]
            if not recs: continue
            em, es, en = ms([r["ecapa"] for r in recs])
            fm, _, _ = ms([r["f0"] for r in recs])
            mc, _, _ = ms([r.get("mcd") for r in recs])
            fr, _, _ = ms([r.get("f0rmse") for r in recs])
            print(f"{ck:<6}{cond:<8}{f'{em:.3f} ± {es:.3f} (n={en})':<26}"
                  f"{(f'{fm:.0f}' if fm else '—'):<9}{(f'{mc:.2f}' if mc else '—'):<8}{(f'{fr:.2f}' if fr else '—'):<10}")
            if cond == "UNSEEN":
                rank[ck] = em

    print("\n=== checkpoint ranking by UNSEEN ECAPA (the overfitting-clean decision) ===")
    for ck, v in sorted(rank.items(), key=lambda x: -(x[1] or -1)):
        print(f"  {v:.3f}   step {ck}")
    print("\n(his own ceiling ~%.3f; base-plain control was ~0.04. MCD/F0-RMSE are SEEN-only cross-checks — overfitting-confounded.)" % np.mean(ceil))

    (EVAL / "_scores.json").write_text(json.dumps(perclip, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
