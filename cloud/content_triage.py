"""
content_triage.py — HINDI content pass (the viability gate).

For each raw Hindi lecture, measure:
  A speaker mix        — pyannote diarization -> per-speaker talk-time
  B which speaker is GGS — ECAPA cosine vs a GGS reference clip -> GGS-only intervals + talk-time
  C how much is HINDI   — Whisper language-ID over ~30s windows of GGS-only speech
=> the number we actually need: of ~4.5 h, how many minutes are clean, solo, HINDI GGS
   (= the trainable pool), vs questioner / Sanskrit-chant / English / (possibly Odia) spans.

Reuses cloud/pod_dataprep.py's diarization + ECAPA cluster-pick VERBATIM (same pyannote/ECAPA
patterns, the torchcodec-bypass in-memory load, the cuDNN LD_LIBRARY_PATH fix); adds only the
language-ID stage. Runs on a data-prep pod (pyannote + faster-whisper); set up with
`bash cloud/pod_setup.sh`.

HONEST CAVEATS (state them, don't hide them):
  - Whisper LID picks ONE language per ~30 s window, so within-window code-switch is invisible;
    windows below --lid-min-prob are counted as "uncertain" (a code-switch/ambiguity proxy).
  - Whisper has NO Odia ("or") label, so his Odia stretches will be MISLABELLED (likely hi/ne/bn/sa).
    Treat this as a coarse Hindi-share + Sanskrit/English router + uncertain-flag — NOT a reliable
    Hindi-vs-Odia separator. For a true Hindi/Odia split add a dedicated Indic audio-LID
    (SpeechBrain VoxLingua107 / AI4Bharat) as a follow-up.
  - ECAPA is English/VoxCeleb-domain; the cosine THRESHOLD may need re-checking on Hindi. Read the
    printed per-cluster cosines and the flags, don't trust a fixed pass/fail.

USAGE
  export HF_TOKEN=hf_xxx                 # accept the pyannote gated repos first
  python cloud/content_triage.py --ref data/hi/refs/ggs_hi_ref.wav               # all data/hi/lectures/*
  python cloud/content_triage.py --ref <ref.wav> --slug ggm-hindi-lecture-03     # one lecture
Outputs: data/hi/lectures/<slug>/content_triage.json  +  data/hi/content_triage.tsv (summary)
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import pod_dataprep as pdp                       # reuse diarization + ECAPA cluster-pick

HI = ROOT / "data" / "hi" / "lectures"

# Whisper language codes -> friendly names (Odia "or" absent from Whisper — see caveat)
WHISPER_NAMES = {"hi": "Hindi", "sa": "Sanskrit", "en": "English", "ur": "Urdu", "bn": "Bengali",
                 "ne": "Nepali", "mr": "Marathi", "pa": "Punjabi", "gu": "Gujarati", "ta": "Tamil",
                 "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "as": "Assamese"}


def resolve_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        for p in ("/workspace/.hf_token", str(Path.home() / ".hf_token")):
            if Path(p).exists():
                tok = Path(p).read_text(encoding="utf-8").strip()
                break
    return tok


def expose_cudnn():
    """Put torch's bundled cuDNN-9 on LD_LIBRARY_PATH before faster-whisper/ctranslate2 load
    (same fix as pod_dataprep.main)."""
    import torch
    d = sorted({os.path.dirname(p) for p in glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cudnn", "lib", "*.so*"))})
    if d:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(d + [os.environ.get("LD_LIBRARY_PATH", "")])


def detect_lang(asr, chunk):
    """Return (lang_code, probability) for a 16k mono float32 chunk. faster-whisper >=1.0 has
    detect_language (guaranteed on the pinned 1.2.0). The transcribe fallback covers older builds:
    language is detected EAGERLY inside transcribe(), so info is valid the moment it returns (no
    need to consume the segment generator)."""
    if hasattr(asr, "detect_language"):
        lang, prob, _ = asr.detect_language(chunk)
        return lang, float(prob)
    _segs, info = asr.transcribe(chunk, language=None, beam_size=1, without_timestamps=True)
    return info.language, float(info.language_probability)


def union_seconds(turns):
    """Wall-clock voiced seconds = union of all speaker turns. pyannote 3.x can emit overlapping
    segments, so sum-per-speaker double-counts cross-talk; the union is the true voiced time."""
    ivs = sorted((t["start"], t["end"]) for t in turns)
    tot, cs, ce = 0.0, None, None
    for s, e in ivs:
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def lid_over_intervals(asr, y, sr, intervals, win, min_prob):
    """Whisper language-ID over fixed windows of GGS-only speech. Returns per-lang seconds,
    uncertain (low-confidence) seconds, total analysed seconds, and per-window records."""
    lang_sec = defaultdict(float)
    uncertain = 0.0
    total = 0.0
    wins = []
    for (s, e) in intervals:
        t = s
        while t < e - 1.0:                        # ignore <1s tails
            w0, w1 = t, min(t + win, e)
            dur = w1 - w0
            t = w1
            if dur < 2.0:                          # too short for reliable LID
                continue
            chunk = y[int(w0 * sr):int(w1 * sr)].astype(np.float32)
            lang, prob = detect_lang(asr, chunk)
            lang_sec[lang] += dur
            total += dur
            if prob < min_prob:
                uncertain += dur
            wins.append({"start": round(w0, 2), "end": round(w1, 2),
                         "lang": lang, "prob": round(prob, 3)})
    return lang_sec, uncertain, total, wins


def process(slug, models, cfg):
    pipe, embed, cosine, centroid, asr = models
    d = HI / slug
    mp3 = d / "audio.mp3"
    assert mp3.exists(), f"missing {mp3} (run scripts/fetch_idt.py first)"
    print(f"\n========== {slug} ==========")
    work = ROOT / "_work" / f"hi_{slug}.wav"
    if work.exists() and work.stat().st_mtime < mp3.stat().st_mtime:
        work.unlink()                            # source re-fetched -> rebuild (to_wav_16k caches on existence only)
    wav16 = pdp.to_wav_16k(mp3, work)

    turns, secs = pdp.diarize(pipe, str(wav16), cfg["min_speakers"], cfg["max_speakers"])
    speech_sec = union_seconds(turns)            # wall-clock voiced time (sum-per-speaker double-counts overlap)
    print(f"  [diarize] {len(turns)} turns, {len(secs)} speakers, {speech_sec/60:.1f}min speech")

    intervals, flags = pdp.verify_ggs(turns, secs, wav16, cfg["ref_clips"],
                                      embed, cosine, centroid, cfg)
    ggs_sec = sum(e - s for s, e in intervals)

    y, sr = librosa.load(str(wav16), sr=16000, mono=True)
    lang_sec, uncertain, lid_total, wins = lid_over_intervals(
        asr, y, sr, intervals, cfg["lid_win"], cfg["lid_min_prob"])
    hindi_sec = lang_sec.get("hi", 0.0)
    lang_min = {WHISPER_NAMES.get(k, k): round(v / 60, 1)
                for k, v in sorted(lang_sec.items(), key=lambda kv: -kv[1])}
    # Hindi share is reported against the LID'd base (what was actually analysed); coverage shows how
    # much of GGS speech got LID'd. hindi_pct_of_ggs (vs full GGS) is kept but is a LOWER bound.
    hindi_pct_lidd = 100 * hindi_sec / lid_total if lid_total > 0 else 0.0
    coverage_pct = 100 * lid_total / ggs_sec if ggs_sec > 0 else 0.0

    rec = {
        "slug": slug,
        "total_speech_min": round(speech_sec / 60, 1),
        "n_speakers": len(secs),
        "speakers_min": {k: round(v / 60, 1) for k, v in sorted(secs.items(), key=lambda kv: -kv[1])},
        "ggs_min": round(ggs_sec / 60, 1),
        "ggs_pct_of_speech": round(100 * ggs_sec / speech_sec, 1) if speech_sec > 0 else 0.0,
        "lang_min_within_ggs": lang_min,
        "hindi_min": round(hindi_sec / 60, 1),
        "hindi_pct_of_lidd": round(hindi_pct_lidd, 1),
        "hindi_pct_of_ggs": round(100 * hindi_sec / ggs_sec, 1) if ggs_sec > 0 else 0.0,
        "lid_coverage_pct": round(coverage_pct, 1),
        "uncertain_min": round(uncertain / 60, 1),
        "caveat": ("Whisper LID has no Odia label, so his Odia stretches are mislabelled (often "
                   "Hindi/Bengali/Nepali) - hindi_min may include Odia. Watch Bengali/Nepali in "
                   "lang_min_within_ggs as an Odia-contamination hint."),
        "flags": flags,
        "ggs_intervals": [[round(s, 2), round(e, 2)] for s, e in intervals],
        "lid_windows": wins,
    }
    d.mkdir(parents=True, exist_ok=True)
    (d / "content_triage.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    other = {k: v for k, v in lang_min.items() if k != "Hindi"}
    print(f"  [content] GGS {rec['ggs_min']:.1f}min ({rec['ggs_pct_of_speech']:.0f}% of speech) | "
          f"Hindi {rec['hindi_min']:.1f}min ({hindi_pct_lidd:.0f}% of LID'd, {coverage_pct:.0f}% coverage) | "
          f"uncertain {rec['uncertain_min']:.1f}min")
    print(f"  [content] other langs within GGS (Bengali/Nepali may be Odia): {other}")
    if flags:
        print(f"  !! flags: {flags}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs="+", required=True,
                    help="clean GGS reference clip(s) for ECAPA (a Hindi clip from the cleanest lecture "
                         "is ideal; an existing English GGS clip also works — ECAPA compares timbre, not words)")
    ap.add_argument("--slug", help="one lecture dir under data/hi/lectures (default: all)")
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=4)
    ap.add_argument("--cosine-thr", type=float, default=0.40)
    ap.add_argument("--cosine-margin", type=float, default=0.10)
    ap.add_argument("--min-turn-sec", type=float, default=1.0)
    ap.add_argument("--lid-win", type=float, default=30.0,
                    help="LID window seconds (Whisper detects one language per window)")
    ap.add_argument("--lid-min-prob", type=float, default=0.5,
                    help="below this language-probability a window is counted 'uncertain' (code-switch/ambiguity)")
    ap.add_argument("--asr-model", default="large-v3")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    tok = resolve_hf_token()
    if not tok:
        sys.exit("No HF token — export HF_TOKEN=... or write /workspace/.hf_token (and accept the pyannote gated repos).")
    for p in a.ref:
        if not Path(p).exists():
            sys.exit(f"reference clip not found: {p}")
    if a.lid_win > 30:
        sys.exit("--lid-win must be <= 30: Whisper LID reads one language from the first 30s of each "
                 "window, so a larger window would mislabel its tail.")

    expose_cudnn()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    cfg = dict(ref_clips=[str(p) for p in a.ref],
               min_speakers=a.min_speakers, max_speakers=a.max_speakers,
               cosine_thr=a.cosine_thr, cosine_margin=a.cosine_margin, min_turn_sec=a.min_turn_sec,
               lid_win=a.lid_win, lid_min_prob=a.lid_min_prob)

    print("loading models (diarizer, ECAPA, faster-whisper) ...")
    pipe = pdp.load_diar_pipeline(tok, device)
    embed, cosine, centroid = pdp.make_embedder(device)
    from faster_whisper import WhisperModel
    asr = WhisperModel(a.asr_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    models = (pipe, embed, cosine, centroid, asr)

    slugs = [a.slug] if a.slug else [p.name for p in sorted(HI.iterdir()) if p.is_dir()]
    if not slugs:
        sys.exit(f"no lectures under {HI} — run scripts/fetch_idt.py first")
    recs = []
    for s in slugs:
        try:
            recs.append(process(s, models, cfg))
        except Exception as e:
            print(f"  !! {s} FAILED: {e}")

    if recs:
        out = ROOT / "data" / "hi" / "content_triage.tsv"
        cols = ["slug", "total_speech_min", "ggs_min", "ggs_pct_of_speech",
                "hindi_min", "hindi_pct_of_lidd", "lid_coverage_pct", "uncertain_min", "flags"]
        with out.open("w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in recs:
                row = dict(r, flags=";".join(r["flags"]) if r["flags"] else "-")
                f.write("\t".join(str(row.get(c)) for c in cols) + "\n")
        clean = [r for r in recs if not r["flags"]]
        tot_ggs = sum(r["ggs_min"] for r in clean)
        tot_hi = sum(r["hindi_min"] for r in clean)
        nflag = len(recs) - len(clean)
        print(f"\n==== TRAINABLE POOL (unflagged) ==== GGS {tot_ggs:.0f}min | Hindi-detected {tot_hi:.0f}min "
              f"across {len(clean)} lecture(s)"
              + (f"  (+{nflag} flagged & excluded - review their cosine/margin)" if nflag else ""))
        print("  NB: 'Hindi' may include Odia (Whisper has no Odia label) - check per-lecture other-langs.")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
