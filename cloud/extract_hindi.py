"""
extract_hindi.py — HINDI data-prep (lean path): turn the confident-Hindi spans content_triage found
into (audio, draft-text) training clips in ONE pass.

Strategy (precision over recall, per the plan): keep only the LID windows Whisper is confident are
Hindi (lang in --langs, prob >= --min-prob; DEFAULT hi+ur, since spoken Hindi/Urdu = Hindustani),
merge them into spans, transcribe each span with faster-whisper (language=hi) to get sentence segments
+ timestamps, and cut ~min..max-sec clips at segment boundaries. The ambiguous 'uncertain'/Odia/English
windows are skipped entirely — we have 100+ min, so clean > more.

Output per lecture:
  data/hi/lectures/<slug>/clips_hi/*.wav
  data/hi/lectures/<slug>/train_hi.jsonl        # {audio, text, lecture, start, end}
The `text` is a Whisper DRAFT — human-correct it before training (that's the remaining effort).

Runs on the same pod as content_triage (GPU faster-whisper). Reuses pod_dataprep.to_wav_16k + cut.

USAGE
  python cloud/extract_hindi.py                     # all lectures, hi+ur
  python cloud/extract_hindi.py --langs hi          # strictest (hi only)
  python cloud/extract_hindi.py --slug ggm-hindi-lecture-03 --min-prob 0.6
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import pod_dataprep as pdp                        # to_wav_16k, cut

HI = ROOT / "data" / "hi" / "lectures"


def merge_windows(wins, langs, min_prob, gap=1.0):
    """Confident-Hindi LID windows -> merged spans (adjacent windows within `gap` seconds joined)."""
    keep = sorted((w["start"], w["end"]) for w in wins
                  if w["lang"] in langs and w["prob"] >= min_prob)
    spans = []
    for s, e in keep:
        if spans and s - spans[-1][1] <= gap:
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
    return spans


def clips_from_segments(segs, min_sec, max_sec):
    """Group whisper segments into (start, end, text) clips, flushing at a segment boundary once the
    clip is >= min_sec (or force-flush at max_sec)."""
    out, cur = [], []
    for seg in segs:
        cur.append(seg)
        dur = cur[-1]["end"] - cur[0]["start"]
        if dur >= min_sec:
            out.append((cur[0]["start"], cur[-1]["end"],
                        " ".join(s["text"].strip() for s in cur).strip()))
            cur = []
        elif cur and (cur[-1]["end"] - cur[0]["start"]) >= max_sec:
            out.append((cur[0]["start"], cur[-1]["end"],
                        " ".join(s["text"].strip() for s in cur).strip()))
            cur = []
    if cur and (cur[-1]["end"] - cur[0]["start"]) >= min_sec:
        out.append((cur[0]["start"], cur[-1]["end"],
                    " ".join(s["text"].strip() for s in cur).strip()))
    return out


def process(slug, asr, cfg):
    d = HI / slug
    tj = d / "content_triage.json"
    if not tj.exists():
        print(f"  skip {slug}: no content_triage.json (run content_triage.py first)")
        return None
    rec = json.loads(tj.read_text(encoding="utf-8"))
    spans = merge_windows(rec.get("lid_windows", []), cfg["langs"], cfg["min_prob"])
    if not spans:
        print(f"  {slug}: no confident-Hindi spans for langs={sorted(cfg['langs'])}")
        return None

    wav16 = pdp.to_wav_16k(d / "audio.mp3", ROOT / "_work" / f"hi_{slug}.wav")
    y, sr = librosa.load(str(wav16), sr=16000, mono=True)
    clips_dir = d / "clips_hi"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.wav"):
        old.unlink()

    rows, idx = [], 1
    for (s, e) in spans:
        chunk = y[int(s * sr):int(e * sr)]
        segs_gen, _ = asr.transcribe(chunk, language="hi", beam_size=5,
                                     vad_filter=True, condition_on_previous_text=False)
        segs = [{"start": s + seg.start, "end": s + seg.end, "text": seg.text}
                for seg in segs_gen if seg.text.strip()]
        for (cs, ce, txt) in clips_from_segments(segs, cfg["min_sec"], cfg["max_sec"]):
            if not txt:
                continue
            dst = clips_dir / f"{slug}_{idx:04d}.wav"
            pdp.cut(wav16, cs, ce, dst)
            rows.append({"audio": f"clips_hi/{dst.name}", "text": txt,
                         "lecture": slug, "start": round(cs, 2), "end": round(ce, 2)})
            idx += 1

    (d / "train_hi.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    tot = sum(r["end"] - r["start"] for r in rows)
    print(f"  {slug}: {len(rows)} clips, {tot/60:.1f}min from {len(spans)} Hindi spans")
    return {"slug": slug, "clips": len(rows), "min": round(tot / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", help="one lecture dir under data/hi/lectures (default: all)")
    ap.add_argument("--langs", default="hi,ur",
                    help="LID labels to keep as Hindi (Hindustani = hi+ur; use 'hi' for strictest)")
    ap.add_argument("--min-prob", type=float, default=0.5, help="min LID confidence to keep a window")
    ap.add_argument("--min-sec", type=float, default=8.0)
    ap.add_argument("--max-sec", type=float, default=18.0)
    ap.add_argument("--asr-model", default="large-v3")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # cuDNN on LD_LIBRARY_PATH before faster-whisper loads (same fix as content_triage/pod_dataprep)
    import torch
    dd = sorted({os.path.dirname(p) for p in glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cudnn", "lib", "*.so*"))})
    if dd:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dd + [os.environ.get("LD_LIBRARY_PATH", "")])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    from faster_whisper import WhisperModel
    asr = WhisperModel(a.asr_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    cfg = dict(langs=set(a.langs.split(",")), min_prob=a.min_prob,
               min_sec=a.min_sec, max_sec=a.max_sec)

    slugs = [a.slug] if a.slug else [p.name for p in sorted(HI.iterdir()) if p.is_dir()]
    recs = [r for r in (process(s, asr, cfg) for s in slugs) if r]
    if recs:
        nc = sum(r["clips"] for r in recs)
        tot = sum(r["min"] for r in recs)
        print(f"\n==== HINDI CLIP SET ==== {nc} clips, {tot:.0f}min across {len(recs)} lecture(s)")
        print("NEXT: review/correct the train_hi.jsonl transcripts (Whisper draft), then group-split -> LoRA train.")


if __name__ == "__main__":
    main()
