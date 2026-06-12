"""
align_cut.py - Step 2b: word-level forced alignment -> excise Sanskrit verse runs
-> cut COHERENT English clips + manifest.

A clip is now a complete idea: it ENDS AT A SENTENCE BOUNDARY (.?!) and groups
~2-3 sentences (~min..max seconds), so it makes sense on its own. Sprinkled
Sanskrit terms stay (his English voice); contiguous Sanskrit runs (verses) are
excised. Each clip gets a small lead-in pad so the first word's onset isn't clipped.

Pipeline (per lecture in data/lectures/<slug>/):
  1. audio.mp3 -> audio.wav (16k mono)
  2. transcript.txt -> transliterate IAST->ASCII                         [D4]
  3. stable-ts: align ASCII text to audio -> per-WORD timestamps (cached -> words.json)
  4. flag each word Sanskrit/English (validated dict detector)           [D3]
  5. excise contiguous Sanskrit runs (>= --excise-run); keep sprinkled terms
  6. group kept words into sentence-boundary clips (~--min-sec..--max-sec)
  7. pad lead-in, cut WAVs (16k mono, trailing silence trimmed) + train.jsonl

SETUP:
  pip install stable-ts                   # pure-Python + torch alignment (NO C++ compiler needed)
RUN:
  python scripts/align_cut.py i-and-mine-and-namabhasa-stage
  # alignment is cached -> re-runs (to tune --min-sec/--max-sec) are instant; --realign to redo.
"""
import argparse, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_transliterate as ct          # transliterate(), is_sanskrit(), load_english()

ROOT = Path(__file__).resolve().parent.parent
LECT = ROOT / "data" / "lectures"

MIN_SEC_DEFAULT = 9.0       # don't end a clip before this -> groups sentences into one idea
MAX_SEC_DEFAULT = 20.0      # force an end by here
HARD_MIN, HARD_MAX = 4.0, 30.0
EXCISE_RUN_DEFAULT = 3      # contiguous Sanskrit words >= this = verse/quote -> excise
LEAD_PAD = 0.15            # secs of lead-in so the first word's onset isn't clipped
TAIL_PAD = 0.15            # secs after the last word (before trailing-silence trim)


def to_wav(mp3: Path) -> Path:
    wav = mp3.with_suffix(".wav")
    if not wav.exists():
        print("[1/5] mp3 -> 16k mono wav ...")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def align(wav: Path, text: str, model_size: str):
    """stable-ts forced alignment of the GIVEN text -> list of {'start','end','text'} per word."""
    import stable_whisper
    print(f"[3/5] Aligning with stable-ts '{model_size}' on CPU "
          f"(a 30-min lecture takes several minutes; cached after)...")
    model = stable_whisper.load_model(model_size)
    result = model.align(str(wav), text, language="en")
    return [{"text": w.word.strip(), "start": w.start, "end": w.end}
            for w in result.all_words() if w.word.strip()]


def excise_zones(words, eng, excise_run):
    """Mark words inside a contiguous Sanskrit run of length >= excise_run."""
    flags = [ct.is_sanskrit(w["text"], eng) for w in words]
    drop = [False] * len(words)
    i = 0
    while i < len(words):
        if flags[i]:
            j = i
            while j < len(words) and flags[j]:
                j += 1
            if j - i >= excise_run:
                for k in range(i, j):
                    drop[k] = True
            i = j
        else:
            i += 1
    return drop


def build_clips(words, drop, min_sec, max_sec):
    """Group non-excised words into clips that END AT A SENTENCE BOUNDARY (.?!),
    each ~min_sec..max_sec long (a coherent 2-3 sentence idea). Break at excised
    (verse) runs. Returns list of (start_idx, end_idx) into `words`."""
    clips, cur = [], []

    def flush():
        if cur:
            s, e = cur[0], cur[-1]
            if HARD_MIN <= words[e]["end"] - words[s]["start"] <= HARD_MAX:
                clips.append((s, e))
        cur.clear()

    for idx, w in enumerate(words):
        if drop[idx]:
            flush(); continue
        cur.append(idx)
        dur = words[cur[-1]]["end"] - words[cur[0]]["start"]
        ends_sentence = w["text"].rstrip().endswith((".", "?", "!"))
        if dur >= max_sec or (ends_sentence and dur >= min_sec):
            flush()
    flush()
    return clips


def cut(src: Path, start: float, end: float, dst: Path):
    # trim silence from BOTH ends to <0.1s (leading trim fixes clips that open with a long pause)
    sr = "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB"
    trim = f"{sr},areverse,{sr},areverse"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-af", trim, "-ac", "1", "-ar", "16000", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", default="i-and-mine-and-namabhasa-stage")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all")
    ap.add_argument("--model", default="tiny", help="stable-ts whisper model: tiny/base/small")
    ap.add_argument("--excise-run", type=int, default=EXCISE_RUN_DEFAULT)
    ap.add_argument("--min-sec", type=float, default=MIN_SEC_DEFAULT)
    ap.add_argument("--max-sec", type=float, default=MAX_SEC_DEFAULT)
    ap.add_argument("--realign", action="store_true", help="redo alignment (ignore cache)")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    d = LECT / args.slug
    mp3 = d / "audio.mp3"
    if not mp3.exists():
        sys.exit(f"Not found: {mp3} (run fetch_lecture.py first)")

    wav = to_wav(mp3)
    print("[2/5] Transliterating transcript IAST -> ASCII ...")
    text = ct.transliterate((d / "transcript.txt").read_text(encoding="utf-8"))

    cache = d / "words.json"
    if cache.exists() and not args.realign:
        words = json.loads(cache.read_text(encoding="utf-8"))
        print(f"[3/5] Loaded cached alignment: {len(words)} words (--realign to redo).")
    else:
        words = align(wav, text, args.model)
        cache.write_text(json.dumps(words), encoding="utf-8")
        print(f"      aligned {len(words)} words (cached -> words.json)")

    print(f"[4/5] Flagging Sanskrit + excising runs >= {args.excise_run} ...")
    eng = ct.load_english()
    drop = excise_zones(words, eng, args.excise_run)
    print(f"      excised {sum(drop)} / {len(words)} words as verse/quote runs")

    clips = build_clips(words, drop, args.min_sec, args.max_sec)
    if args.max_clips:
        clips = clips[: args.max_clips]

    clips_dir = d / "clips"; clips_dir.mkdir(exist_ok=True)
    for old in clips_dir.glob("*.wav"):           # clear stale clips from prior runs
        old.unlink()
    manifest = d / "train.jsonl"
    print(f"[5/5] Cutting {len(clips)} coherent English clips -> {manifest.name} ...")
    durs = []
    with manifest.open("w", encoding="utf-8") as mf:
        for i, (s, e) in enumerate(clips, 1):
            prev_end = words[s - 1]["end"] if s > 0 else 0.0
            nxt = words[e + 1]["start"] if e + 1 < len(words) else words[e]["end"] + 1
            start = max(words[s]["start"] - LEAD_PAD, prev_end, 0.0)   # lead-in, don't enter prev word
            end = words[e]["end"] + min(TAIL_PAD, max(0.0, nxt - words[e]["end"]))
            txt = " ".join(words[k]["text"] for k in range(s, e + 1))
            dst = clips_dir / f"{args.slug}_{i:04d}.wav"
            cut(wav, start, end, dst)
            durs.append(end - start)
            mf.write(json.dumps({"audio": f"clips/{dst.name}", "text": txt,
                                 "duration": round(end - start, 2)}, ensure_ascii=False) + "\n")

    if durs:
        avg = sum(durs) / len(durs)
        print(f"\nDONE. {len(durs)} clips ({sum(durs)/60:.1f} min)  "
              f"len: min={min(durs):.1f} avg={avg:.1f} max={max(durs):.1f}s -> {clips_dir}")
    print(f"Manifest -> {manifest}")
    print("NEXT: re-listen to a few clips — now full 2-3 sentence ideas, first words intact. "
          "Tune --min-sec/--max-sec if you want longer/shorter.")


if __name__ == "__main__":
    main()
