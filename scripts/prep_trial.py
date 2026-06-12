"""
prep_trial.py - Trial data-prep for the GGS VoxCPM2 LoRA fine-tune.

Takes ONE YouTube lecture -> downloads audio -> transcribes with timestamps ->
packs phrase segments into 5-15s clips -> cuts WAVs -> writes a JSONL manifest.

This is the "fast iteration on a small representative subset" step (fastbook Ch7):
validate the whole pipeline on ~10 clips BEFORE processing all 5+ hours.

--------------------------------------------------------------------------------
ONE-TIME SETUP (Windows, CPU is fine for the trial):
    # ffmpeg (pick one):
    winget install Gyan.FFmpeg            # or: choco install ffmpeg
    # python deps:
    pip install yt-dlp faster-whisper
--------------------------------------------------------------------------------
RUN:
    python scripts/prep_trial.py "https://www.youtube.com/watch?v=XXXX" --max-clips 10

OUTPUT:
    data/trial/raw/<id>.wav          full downloaded lecture audio (16k mono)
    data/trial/clips/ggs_0001.wav    the cut training clips
    data/trial/trial.jsonl           manifest: {"audio": ..., "text": ...} per clip

Then LISTEN to a few clips and open trial.jsonl to confirm text matches audio.
Recipe enforced (from VoxCPM fine-tuning guide): clips 3-30s (target 5-15s),
trailing silence trimmed, English-only is YOUR manual review pass after this runs.
"""
import argparse, json, subprocess, sys
from pathlib import Path

# ---- tunables (match finetune-plan.md recipe) -------------------------------
TARGET_MIN, TARGET_MAX = 5.0, 15.0     # preferred clip length window (seconds)
HARD_MIN,  HARD_MAX    = 3.0, 30.0     # guide's absolute bounds; outside -> drop
WHISPER_MODEL          = "small"        # "small" = good CPU speed/accuracy for trial; "medium" = better
LANG                   = "en"           # force English decoding (we want English-only clips)

ROOT      = Path(__file__).resolve().parent.parent
OUT_DIR   = ROOT / "data" / "trial"
RAW_DIR   = OUT_DIR / "raw"
CLIPS_DIR = OUT_DIR / "clips"
MANIFEST  = OUT_DIR / "trial.jsonl"


def run(cmd: list[str]) -> None:
    """Run a shell command, streaming output; raise on failure."""
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_audio(url: str) -> Path:
    """Download best audio from YouTube as a 16k mono WAV via yt-dlp + ffmpeg."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(RAW_DIR / "%(id)s.%(ext)s")
    run([sys.executable, "-m", "yt_dlp",
         "-x", "--audio-format", "wav",            # extract audio -> wav
         "--postprocessor-args", "-ac 1 -ar 16000",  # mono, 16kHz (VoxCPM2 encoder rate)
         "-o", out_tmpl, url])
    wavs = sorted(RAW_DIR.glob("*.wav"))
    if not wavs:
        sys.exit("No WAV produced by yt-dlp - check the URL / ffmpeg install.")
    return wavs[-1]


def transcribe(wav: Path):
    """Return faster-whisper segments: list of (start, end, text)."""
    from faster_whisper import WhisperModel
    print(f"[2/4] Transcribing with faster-whisper '{WHISPER_MODEL}' (CPU int8)...")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), language=LANG, vad_filter=True)
    segs = [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    print(f"      got {len(segs)} raw phrase segments")
    return segs


def pack_clips(segs):
    """Greedily merge consecutive segments into clips within the target window."""
    clips, cur_text, cur_start, cur_end = [], [], None, None
    for start, end, text in segs:
        if cur_start is None:
            cur_start, cur_end, cur_text = start, end, [text]
            continue
        # if adding this segment keeps us <= TARGET_MAX, extend the current clip
        if end - cur_start <= TARGET_MAX:
            cur_end, _ = end, cur_text.append(text)
        else:
            clips.append((cur_start, cur_end, " ".join(cur_text)))
            cur_start, cur_end, cur_text = start, end, [text]
    if cur_start is not None:
        clips.append((cur_start, cur_end, " ".join(cur_text)))
    # keep only clips within the guide's hard bounds
    kept = [(s, e, t) for (s, e, t) in clips if HARD_MIN <= (e - s) <= HARD_MAX]
    print(f"[3/4] Packed into {len(kept)} clips in [{HARD_MIN},{HARD_MAX}]s "
          f"(target {TARGET_MIN}-{TARGET_MAX}s)")
    return kept


def cut_clip(src: Path, start: float, end: float, dst: Path) -> None:
    """Cut [start,end] from src, trim trailing silence to <0.5s, write 16k mono WAV."""
    # areverse + silenceremove + areverse = trim silence from the END only
    trim = ("areverse,silenceremove=start_periods=1:start_silence=0.1:"
            "start_threshold=-40dB,areverse")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-af", trim, "-ac", "1", "-ar", "16000", str(dst)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube URL of ONE GGS English lecture")
    ap.add_argument("--max-clips", type=int, default=10, help="stop after N clips (trial)")
    args = ap.parse_args()

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Downloading audio...")
    wav = download_audio(args.url)

    segs = transcribe(wav)
    clips = pack_clips(segs)[: args.max_clips]

    print(f"[4/4] Cutting {len(clips)} clips + writing manifest...")
    with MANIFEST.open("w", encoding="utf-8") as mf:
        for i, (start, end, text) in enumerate(clips, 1):
            dst = CLIPS_DIR / f"ggs_{i:04d}.wav"
            cut_clip(wav, start, end, dst)
            # path is relative to the manifest's folder, for portability
            line = {"audio": f"clips/{dst.name}", "text": text,
                    "duration": round(end - start, 2)}
            mf.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nDONE. {len(clips)} clips -> {CLIPS_DIR}")
    print(f"Manifest -> {MANIFEST}")
    print("NEXT: listen to a few clips, open trial.jsonl, and verify text matches "
          "audio + that each clip is clean English (drop any code-switched ones by hand).")


if __name__ == "__main__":
    main()
