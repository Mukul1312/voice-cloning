"""
probe_hi_refs.py — VoxCPM2 Hindi go/no-go probe over MULTIPLE hand-picked references.

Reads a probe_refs.jsonl (exported from the review gallery via "⭐ Download probe refs": rows
{audio, text, lecture, start, end}, i.e. the user-corrected transcript per starred clip) and, for each
reference, generates a few Hindi test sentences in his voice with BASE VoxCPM2 (no LoRA), hifi cloning
(prompt_text = the ref's corrected Devanagari). Ear-check: does base VoxCPM2 render clean, in-his-voice
Hindi? -> the VoxCPM2-vs-IndicF5 decision.

Runs from the VoxCPM2 venv on the pod (pod_infer.sh env):
  /workspace/venv/bin/python cloud/probe_hi_refs.py --refs /workspace/probe_refs.jsonl

Outputs: /workspace/out_probe_hi/{refid}__{sentid}__s{seed}.wav  (self-describing; resumable).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf
from voxcpm.core import VoxCPM

ROOT = Path(__file__).resolve().parent.parent
HI = ROOT / "data" / "hi" / "lectures"
BASE = "/workspace/VoxCPM2"
OUT = "/workspace/out_probe_hi"

# Hindi test sentences (Devanagari) — his devotional register + a numbers/date read.
# Author-provided best-effort; edit to taste (they're just what we ask the clone to SAY).
SENTS = [
    ("shelter", "कृष्ण ही हमारे एकमात्र आश्रय हैं।"),
    ("guru",    "गुरु की कृपा के बिना कोई इस भवसागर को पार नहीं कर सकता।"),
    ("prayer",  "हे प्रभु, इस सेवक पर अपनी कृपा बरसाइए और इसे शक्ति दीजिए।"),
]
SEEDS = [42, 123, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="probe_refs.jsonl exported from the review gallery")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--denoise", action="store_true", help="run VoxCPM's built-in denoiser on each reference")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    refs = [json.loads(l) for l in Path(a.refs).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not refs:
        sys.exit(f"no refs in {a.refs}")
    os.makedirs(OUT, exist_ok=True)

    print(f"loading base VoxCPM2 (no LoRA): {len(refs)} refs x {len(SENTS)} sents x {len(a.seeds)} seeds",
          file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False)
    sr = m.tts_model.sample_rate

    for i, r in enumerate(refs, 1):
        wav = HI / r["lecture"] / r["audio"]           # data/hi/lectures/<lecture>/clips_hi/xxx.wav
        refid = f"r{i:02d}"
        if not wav.exists():
            print(f"  {refid} MISSING wav: {wav}", file=sys.stderr)
            continue
        print(f"  {refid}: {r['audio']}  text='{r['text'][:40]}...'", file=sys.stderr)
        for (sid, text) in SENTS:
            for s in a.seeds:
                out = f"{OUT}/{refid}__{sid}__s{s}.wav"
                if os.path.exists(out):
                    continue
                w = m.generate(text=text, prompt_wav_path=str(wav), prompt_text=r["text"],
                               denoise=a.denoise, seed=s)
                sf.write(out, w, sr)
                print(f"wrote {os.path.basename(out)}", file=sys.stderr)

    # small manifest mapping refid -> source clip, so we know which ref made which take
    (Path(OUT) / "refs_map.json").write_text(
        json.dumps({f"r{i:02d}": {"audio": r["audio"], "lecture": r["lecture"], "text": r["text"]}
                    for i, r in enumerate(refs, 1)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE — pull /workspace/out_probe_hi/ and ear-check (refs_map.json says which ref is which).")


if __name__ == "__main__":
    main()
