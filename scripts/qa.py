"""
qa.py - metrics-based QA over a lecture's cut clips. Scales to thousands of clips:
cheap metrics flag the bad ones; we only eyeball a small gallery of flagged + controls.

For data/lectures/<slug>/ it:
  - scores every clip in train.jsonl:
      * words/sec  (text words / duration)  -> low = alignment-failure / text-audio mismatch
      * lead/trail silence (from audio RMS)  -> excess = bad cut
  - writes:
      * qa_report.tsv        (every clip + metrics + PASS/FAIL + reason)
      * train.clean.jsonl    (PASSing clips only = the curated training manifest)
      * qa_gallery.png       (mel-spectrograms of the worst-scoring + control clips, to calibrate)
  - prints a summary.

RUN (inside the venv):
  ./.venv/Scripts/python.exe scripts/qa.py i-and-mine-and-namabhasa-stage
  # tune: --min-wps 0.9   --max-lead 0.6
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter
import numpy as np
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SIL_DB = -40            # frames below this (rel. to clip peak) count as silence


def silence_edges(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    voiced = np.where(db > SIL_DB)[0]
    if len(voiced) == 0:
        return None
    t = librosa.frames_to_time(voiced, sr=sr, hop_length=256)
    return float(t[0]), float(len(y) / sr - t[-1])      # (lead_sil, trail_sil)


def gallery(d, slug, rows, picks, dst):
    # one column = big, readable panels (a tall image, but each spectrogram is legible)
    fig, axs = plt.subplots(len(picks), 1, figsize=(13, 2.4 * len(picks)))
    axs = np.array(axs).reshape(-1)
    for ax in axs:
        ax.axis("off")
    for ax, (ci, tag) in zip(axs, picks):
        wav = d / "clips" / f"{slug}_{ci:04d}.wav"
        if not wav.exists():
            continue
        y, sr = librosa.load(str(wav), sr=16000)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
        ax.axis("on")
        librosa.display.specshow(mel, sr=sr, x_axis="time", ax=ax, cmap="magma")
        r = rows[ci - 1]
        ax.set_title(f"clip {ci} [{tag}]  {r['duration']}s/{len(r['text'].split())}w", fontsize=9)
        ax.set_yticks([])
    fig.tight_layout(); fig.savefig(str(dst), dpi=95); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", default="i-and-mine-and-namabhasa-stage")
    ap.add_argument("--min-wps", type=float, default=1.0)
    ap.add_argument("--max-lead", type=float, default=0.6)
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    d = ROOT / "data" / "lectures" / args.slug
    rows = [json.loads(l) for l in open(d / "train.jsonl", encoding="utf-8")]

    rep, passed = [], []
    print(f"Scoring {len(rows)} clips (loading audio)...")
    for i, r in enumerate(rows, 1):
        words, dur = len(r["text"].split()), r["duration"]
        wps = words / dur if dur else 0.0
        reasons = []
        if wps < args.min_wps:
            reasons.append(f"low_wps({wps:.2f})")
        lead = -1.0
        wav = d / "clips" / Path(r["audio"]).name
        if wav.exists():
            y, sr = librosa.load(str(wav), sr=16000)
            se = silence_edges(y, sr)
            if se is None:
                reasons.append("all_silence")
            else:
                lead = round(se[0], 2)
                if se[0] > args.max_lead:
                    reasons.append(f"lead_sil({se[0]:.2f})")
        ok = not reasons
        rep.append({"clip": i, "dur": dur, "words": words, "wps": round(wps, 2),
                    "lead": lead, "status": "PASS" if ok else "FAIL",
                    "reasons": ";".join(reasons)})
        if ok:
            passed.append(r)

    # artifacts
    (d / "train.clean.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in passed), encoding="utf-8")
    with (d / "qa_report.tsv").open("w", encoding="utf-8") as f:
        f.write("clip\tdur\twords\twps\tlead_sil\tstatus\treasons\n")
        for x in rep:
            f.write(f"{x['clip']}\t{x['dur']}\t{x['words']}\t{x['wps']}\t{x['lead']}\t{x['status']}\t{x['reasons']}\n")

    # gallery: 6 worst-by-wps (to calibrate the cutoff) + 3 PASS controls
    worst = sorted(rep, key=lambda x: x["wps"])[:6]
    ctrls = [x for x in rep if x["status"] == "PASS"][len(passed)//2: len(passed)//2 + 3]
    picks = [(x["clip"], x["reasons"] or "FAIL") for x in worst] + [(x["clip"], "ctrl") for x in ctrls]
    gallery(d, args.slug, rows, picks, d / "qa_gallery.png")

    # summary
    nfail = sum(x["status"] == "FAIL" for x in rep)
    c = Counter()
    for x in rep:
        if x["status"] == "FAIL":
            for rs in x["reasons"].split(";"):
                c[rs.split("(")[0]] += 1
    wpsv = sorted(x["wps"] for x in rep)
    print(f"\n{len(rows)} clips -> PASS {len(passed)} / FAIL {nfail}   (--min-wps {args.min_wps})")
    print(f"fail reasons: {dict(c)}")
    print(f"wps distribution: min={wpsv[0]} p10={wpsv[len(wpsv)//10]} median={wpsv[len(wpsv)//2]} max={wpsv[-1]}")
    print(f"clean manifest -> train.clean.jsonl ({len(passed)} clips)")
    print(f"report -> qa_report.tsv   gallery -> qa_gallery.png")


if __name__ == "__main__":
    main()
