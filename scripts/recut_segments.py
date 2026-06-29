"""
recut_segments.py — turn a hand-reviewed resegment_export.json into a clean TTS dataset.

Cuts each KEEP segment from the full lecture wav at the human-set boundaries (16k mono),
normalizes the text (IAST->ASCII, strip [Laugh]/[Hmm] annotations), and writes train.jsonl.

Usage:
  python scripts/recut_segments.py \
      --export data/out/lecture1/resegment_export.json \
      --audio  data/out/lecture1/lecture_full.wav \
      --out    data/out/lecture1/final \
      --slug   ggs_l1
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from classify_transliterate import transliterate          # IAST -> lossy ASCII (verses kept)


def clean_text(t: str) -> str:
    t = transliterate(t)                                   # diacritics -> ASCII
    t = re.sub(r"\[\s*[Ll]augh[^\]]*\]", " ", t)           # [Laugh] -> drop (not spoken words)
    t = re.sub(r"\[\s*[Hh]+\s*m+[^\]]*\]", " hmm ", t)     # [Hmm]/[Hm] -> hmm (he says it)
    t = re.sub(r"\[[^\]]*\]", " ", t)                      # any other [annotation] -> drop
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("…", "...").replace("–", "-")
    t = re.sub(r"\s+([,.?!;:])", r"\1", t)                 # no space before punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


def cut(audio: Path, start: float, end: float, dst: Path):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(audio),
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-ac", "1", "-ar", "16000", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", default="ggs")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    keeps = [d for d in json.loads(Path(args.export).read_text(encoding="utf-8")) if d["decision"] == "keep"]
    outd = Path(args.out); clips = outd / "clips"; clips.mkdir(parents=True, exist_ok=True)
    for old in clips.glob("*.wav"): old.unlink()
    audio = Path(args.audio)
    rows, leftover = [], []
    for i, d in enumerate(keeps, 1):
        dst = clips / f"{args.slug}_{i:04d}.wav"
        cut(audio, float(d["start"]), float(d["end"]), dst)
        txt = clean_text(d["text"])
        if any(ord(c) > 127 for c in txt): leftover.append((i, txt))
        rows.append({"audio": f"clips/{dst.name}", "text": txt, "duration": round(float(d["end"]) - float(d["start"]), 2)})
    (outd / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    durs = [r["duration"] for r in rows]
    print(f"wrote {len(rows)} clips -> {clips}  ({sum(durs)/60:.1f} min)")
    print(f"train.jsonl -> {outd/'train.jsonl'}")
    print(f"non-ASCII remaining after clean: {leftover if leftover else 'none'}")


if __name__ == "__main__":
    main()
