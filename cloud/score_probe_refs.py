"""
score_probe_refs.py — reference-selection scorer (ECAPA cosine-to-centroid), practised on the probe.

The English project's dominant lever was picking the reference by ECAPA cosine-to-centroid
(representativeness = how him-typical a clip is), and it PREDICTED downstream output identity. This
scorer checks that on Hindi:

  centroid = his Hindi voice, from all clips_hi (the 4 refs excluded)
  (a) representativeness : each ref clip's own cosine to the centroid
  (b) output identity    : mean cosine of the takes that ref generated (out_probe_hi) to the centroid
  -> rank refs by each; do (a) and (b) agree?

Later: run (a) over ALL 210 CORRECTED clips to pick the production reference.

Runs on the content pod (system python: speechbrain ECAPA). Reuses pod_dataprep.make_embedder.
  python cloud/score_probe_refs.py --refs /workspace/probe_refs.jsonl
"""
import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import pod_dataprep as pdp                      # make_embedder (ECAPA)

HI = ROOT / "data" / "hi" / "lectures"
TAKES = Path("/workspace/out_probe_hi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="probe_refs.jsonl (the 4 starred refs)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embed, cosine, centroid = pdp.make_embedder(dev)

    allclips = sorted(HI.glob("*/clips_hi/*.wav"))
    refs = [json.loads(l) for l in Path(a.refs).read_text(encoding="utf-8").splitlines() if l.strip()]
    ref_path = {i + 1: str(HI / r["lecture"] / r["audio"]) for i, r in enumerate(refs)}
    ref_set = set(ref_path.values())
    pool = [str(p) for p in allclips if str(p) not in ref_set]
    print(f"centroid from {len(pool)} Hindi clips (excluding the {len(ref_set)} refs)", file=sys.stderr)
    cen = centroid(pool)

    # (a) representativeness of each ref
    reps = {}
    print("\n== (a) reference representativeness — cosine to his-centroid (higher = more him-typical) ==")
    for i, r in enumerate(refs, 1):
        rid = f"r{i:02d}"
        c = cosine(embed(ref_path[i]), cen)
        reps[rid] = c
        print(f"  {rid}  {Path(r['audio']).name:34s} rep={c:+.3f}")

    # (b) output identity per ref (its generated takes)
    pat = re.compile(r"^(r\d+)__[a-z]+__s\d+\.wav$")
    byref = defaultdict(list)
    for w in sorted(TAKES.glob("r*.wav")):
        m = pat.match(w.name)
        if m:
            byref[m.group(1)].append(cosine(embed(str(w)), cen))
    outs = {rid: st.mean(v) for rid, v in byref.items()}
    print("\n== (b) output identity — mean cosine of each ref's takes to centroid ==")
    for rid in sorted(outs):
        print(f"  {rid}  out={outs[rid]:+.3f}  (n={len(byref[rid])})")

    # ranking + agreement
    print("\n== ranking ==")
    print("  by representativeness:", "  >  ".join(f"{r}({reps[r]:+.3f})" for r in sorted(reps, key=reps.get, reverse=True)))
    if outs:
        print("  by output identity   :", "  >  ".join(f"{r}({outs[r]:+.3f})" for r in sorted(outs, key=outs.get, reverse=True)))
        best_rep, best_out = max(reps, key=reps.get), max(outs, key=outs.get)
        print(f"\n  best by representativeness = {best_rep} | best by output = {best_out} "
              f"-> {'AGREE (metric validated)' if best_rep == best_out else 'DIFFER (inspect)'}")
    print("\nNEXT: rerun (a) over all 210 corrected clips to pick the production reference.")


if __name__ == "__main__":
    main()
