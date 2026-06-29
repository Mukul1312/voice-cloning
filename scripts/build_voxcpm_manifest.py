"""
build_voxcpm_manifest.py — turn final/train.jsonl into a VoxCPM2-format training manifest.

Per the VoxCPM fine-tuning guide: adds ref_audio to ~40% of samples (a random other clip from the
same speaker), keeps duration, and rewrites paths to absolute POD paths so the trainer resolves them.
Also prints a clean English clip to use as the INFERENCE reference (--prompt_audio at test time).

Usage:
  python scripts/build_voxcpm_manifest.py \
      --final data/out/lecture1/final \
      --pod-prefix /workspace/voice-cloning/data/out/lecture1/final \
      --ref-frac 0.4
"""
import argparse, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="data/out/lecture1/final")
    ap.add_argument("--pod-prefix", default="/workspace/voice-cloning/data/out/lecture1/final")
    ap.add_argument("--ref-frac", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    fdir = (ROOT / args.final) if not Path(args.final).is_absolute() else Path(args.final)
    rows = [json.loads(l) for l in (fdir / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pod = args.pod_prefix.rstrip("/")
    rng = random.Random(args.seed)
    files = [r["audio"] for r in rows]   # e.g. clips/ggs_l1_0001.wav
    apath = lambda rel: f"{pod}/{rel}"

    out = []
    n_ref = 0
    for r in rows:
        o = {"audio": apath(r["audio"]), "text": r["text"], "duration": r["duration"]}
        if rng.random() < args.ref_frac:
            ref = rng.choice([f for f in files if f != r["audio"]])
            o["ref_audio"] = apath(ref); n_ref += 1
        out.append(o)
    dst = fdir / "train.voxcpm.jsonl"
    dst.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out), encoding="utf-8")
    print(f"wrote {len(out)} samples -> {dst}  ({n_ref} with ref_audio, {100*n_ref/len(out):.0f}%)")

    # pick a clean English clip as inference reference: 6-12s, English-dense, no filler/brackets
    eng = set((ROOT / "data" / "english_words.txt").read_text(encoding="utf-8").split())
    tok = re.compile(r"[a-z']+")
    def efrac(t):
        w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
    cand = [r for r in rows if 6 <= r["duration"] <= 12 and "hmm" not in r["text"].lower()
            and "[" not in r["text"] and efrac(r["text"]) >= 0.9]
    cand.sort(key=lambda r: -len(r["text"]))
    if cand:
        ref = cand[0]
        print("\nINFERENCE reference clip (use as --prompt_audio + --prompt_text):")
        print(f"  audio : {apath(ref['audio'])}")
        print(f"  text  : {ref['text']}")


if __name__ == "__main__":
    main()
