"""
build_probe_gallery.py — audition page for the VoxCPM2 Hindi probe takes.

Reads /workspace/out_probe_hi (rNN__sentid__sSEED.wav + refs_map.json) + the source reference clips, and
emits voice_lab/probe_hi/ : per reference, HIS REAL clip (to compare identity) + the generated Hindi takes
(each test sentence × seeds). Served by the pod's http.server -> browse http://localhost:8000/probe_hi/ over
the tunnel. Runs on the pod (stdlib only).

  python scripts/build_probe_gallery.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAKES = Path("/workspace/out_probe_hi")
HI = ROOT / "data" / "hi" / "lectures"
OUT = ROOT / "voice_lab" / "probe_hi"
AUD = OUT / "audio"

SENTS = {
    "shelter": "कृष्ण ही हमारे एकमात्र आश्रय हैं।",
    "guru": "गुरु की कृपा के बिना कोई इस भवसागर को पार नहीं कर सकता।",
    "prayer": "हे प्रभु, इस सेवक पर अपनी कृपा बरसाइए और इसे शक्ति दीजिए।",
}

HEAD = """<!doctype html><html lang="hi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>VoxCPM2 Hindi probe</title>
<style>
 body{margin:0;background:#12100f;color:#f3eee9;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:820px;margin:0 auto;padding:20px}
 h1{font-size:20px;color:#c9a24a} .note{color:#a89f97;font-size:13px;margin:0 0 20px}
 .ref{background:#1d1a18;border:1px solid #332e2a;border-radius:12px;padding:14px 16px;margin:0 0 18px}
 .ref h2{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#c9a24a;margin:0 0 8px}
 .rt{font:16px/1.5 'Noto Sans Devanagari',serif;color:#d8c7a0;margin:6px 0 12px}
 .sent{border-top:1px solid #2a2521;padding:10px 0}
 .stext{font:16px/1.5 'Noto Sans Devanagari',serif;margin:0 0 6px}
 .take{display:inline-flex;align-items:center;gap:6px;margin:0 12px 8px 0;color:#a89f97;font-size:12px}
 audio{height:32px;vertical-align:middle}
 audio.real{width:100%;height:38px}
</style></head><body><div class="wrap">
<h1>VoxCPM2 — Hindi probe (base model, no LoRA)</h1>
<p class="note">Each block: the REAL reference clip (his voice) on top, then the generated Hindi takes for
3 test sentences × 3 seeds. Ear-check: does the generated Hindi sound (a) like <b>him</b> and (b) clean +
intelligible? That's the VoxCPM2-vs-IndicF5 go/no-go.</p>
"""
FOOT = "</div></body></html>"


def main():
    if not TAKES.exists():
        sys.exit("no /workspace/out_probe_hi — run cloud/probe_hi_refs.py first")
    rmap = {}
    if (TAKES / "refs_map.json").exists():
        rmap = json.loads((TAKES / "refs_map.json").read_text(encoding="utf-8"))
    AUD.mkdir(parents=True, exist_ok=True)
    for old in AUD.glob("*.wav"):
        old.unlink()

    pat = re.compile(r"^(r\d+)__([a-z]+)__s(\d+)\.wav$")
    takes = {}
    for w in sorted(TAKES.glob("r*.wav")):
        m = pat.match(w.name)
        if not m:
            continue
        rid, sid, seed = m.group(1), m.group(2), m.group(3)
        shutil.copy(w, AUD / w.name)
        takes.setdefault(rid, {}).setdefault(sid, []).append((seed, w.name))

    html = [HEAD]
    for rid in sorted(takes):
        info = rmap.get(rid, {})
        html.append(f'<div class="ref"><h2>{rid} · reference (his real voice)</h2>')
        src = HI / info.get("lecture", "") / info.get("audio", "")
        if info and src.exists():
            dst = f"ref_{rid}.wav"
            shutil.copy(src, AUD / dst)
            html.append(f'<audio class="real" controls preload="none" src="audio/{dst}"></audio>')
        if info.get("text"):
            html.append(f'<div class="rt">{info["text"]}</div>')
        for sid in sorted(takes[rid]):
            html.append(f'<div class="sent"><div class="stext">{SENTS.get(sid, sid)}</div>')
            for seed, fn in sorted(takes[rid][sid]):
                html.append(f'<span class="take">s{seed} <audio controls preload="none" src="audio/{fn}"></audio></span>')
            html.append('</div>')
        html.append('</div>')
    html.append(FOOT)
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    n = sum(len(v) for r in takes.values() for v in r.values())
    print(f"wrote {OUT / 'index.html'}  ({n} takes across {len(takes)} refs)")


if __name__ == "__main__":
    main()
