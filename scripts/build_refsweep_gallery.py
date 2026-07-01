"""
build_refsweep_gallery.py — EAR-CHECK the reference-sweep winner. Hear the LoRA-150 clone under each
reference (0081-winner / 0086-calm / 0112-old-baseline) on the two UNSEEN sentences, next to the real
reference clips + a real GGS anchor, with per-clip ECAPA cos-to-his-centroid.
The question for the ear: does 0081 (metric winner, 0.806) sound like HIM *and* not too shouty?
Audio-only (no librosa) -> robust. Output -> voice_lab/refsweep/ (servable, not under data/).
Run: .venv/Scripts/python.exe scripts/build_refsweep_gallery.py
"""
import json
import re
import shutil
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data/out/lecture1/final"
CLIPS = FINAL / "clips"
CLIPS_DN = FINAL / "clips_dn"
SWEEP = ROOT / "data/out/lecture1/refsweep"
OUT = ROOT / "voice_lab/refsweep"
AUD = OUT / "audio"
DEV = "cpu"

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Reference sweep - ear check</title>
<style>
 :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;--win:#46c08a;--calm:#5aa9ff;--old:#ff9d4a;--real:#c78bff}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:1000px;margin:0 auto;padding:20px 16px 80px}h1{font-size:20px;margin:0 0 4px}
 .sub{color:var(--mut);max-width:760px;margin:0 0 16px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:0 0 12px}
 .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.grow{flex:1;min-width:180px}
 audio{width:100%;height:34px}.pill{font-weight:700;padding:1px 8px;border-radius:20px;font-size:12px;color:#0f1115}
 .cos{font-family:ui-monospace,monospace;color:var(--mut);font-size:12px}
 h2{font-size:15px;margin:16px 0 8px}.q{color:var(--mut);font-size:12px;margin:2px 0 8px}
 .cols{display:flex;gap:12px;flex-wrap:wrap}.col{flex:1 1 300px;border:1px solid var(--line);border-radius:10px;padding:8px 10px}
 .col.win{border-color:var(--win)}.col.calm{border-color:var(--calm)}.col.old{border-color:var(--old)}
 .lab{font-weight:700;font-size:13px;margin:0 0 6px}.seed{margin:0 0 6px}.tx{color:var(--ink);font-style:italic;margin:0 0 8px}
</style></head><body><div class="wrap">
 <h1>&#127911; Reference sweep &mdash; ear check</h1>
 <p class="sub">Metric winner is <b style="color:var(--win)">0081</b> (LoRA-150, unseen ECAPA <b>0.806</b>, ceiling <span id="ceil"></span>). Your job: does 0081 sound like <b>him</b> &mdash; and is the <b>shout</b> bleeding into the output (too loud/animated)? If it is, <b style="color:var(--calm)">0086</b> (0.779, calmer) is the graceful pick. Compare vs the real anchor + the reference clips that steer each clone.</p>
 <div class="card"><div class="row"><span class="pill" style="background:var(--real)">REAL GGS</span>
   <span class="grow"><audio controls preload="none" id="anchor"></audio></span><span class="cos" id="anchorcos"></span></div></div>
 <h2>The three references (what steers the clone)</h2><div id="refs"></div>
 <h2>The clone outputs &mdash; LoRA-150, unseen sentences</h2>
 <p class="q">Same model, same sentences, same seeds &mdash; only the reference clip differs. A/B the columns.</p>
 <div id="sents"></div>
</div>
<script>
const D=/*__DATA__*/;const $=s=>document.querySelector(s);
const PILL={win:"var(--win)",calm:"var(--calm)",old:"var(--old)"};
$("#ceil").textContent=D.ceiling;$("#anchor").src=D.anchor.url;$("#anchorcos").textContent="cos "+D.anchor.cos;
$("#refs").innerHTML=D.refs.map(r=>`<div class="card"><div class="row"><span class="pill" style="background:${PILL[r.cls]}">${r.label}</span><span class="grow"><audio controls preload="none" src="${r.url}"></audio></span><span class="cos">clip cos ${r.cos}</span></div></div>`).join("");
$("#sents").innerHTML=D.sents.map(s=>`<div class="card"><div class="tx">&ldquo;${s.text}&rdquo;</div><div class="cols">${
 s.refs.map(rf=>`<div class="col ${rf.cls}"><div class="lab" style="color:${PILL[rf.cls]}">${rf.label} &middot; mean ${rf.mean}</div>${
  rf.seeds.map(sd=>`<div class="seed"><div class="cos">seed ${sd.seed} &middot; cos ${sd.cos}</div><audio controls preload="none" src="${sd.url}"></audio></div>`).join("")}</div>`).join("")
}</div></div>`).join("");
</script></body></html>
"""

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                     savedir=str(ROOT / ".venv/spkrec"), run_opts={"device": DEV},
                                     local_strategy=LocalStrategy.COPY)


def load16k(p):
    y, sr = sf.read(str(p), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        y = resample_poly(y, 16000 // g, sr // g).astype("float32")
    return y


@torch.no_grad()
def embed(p):
    w = torch.from_numpy(np.ascontiguousarray(load16k(p))).unsqueeze(0)
    return F.normalize(clf.encode_batch(w).squeeze(0).squeeze(0), dim=0)


def cos(a, b):
    return float(torch.dot(a, b))


rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCL = {"ggs_l1_0112", "ggs_l1_0114", "ggs_l1_0086", "ggs_l1_0081"}
sel = [Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
ref = F.normalize(torch.stack([embed(CLIPS / n) for n in sel[:20]]).mean(0), dim=0)
ceil = round(float(np.mean([cos(embed(CLIPS / n), ref) for n in sel[20:30]])), 3)

AUD.mkdir(parents=True, exist_ok=True)
def put(src, name):
    dst = AUD / name
    if not dst.exists():
        shutil.copyfile(src, dst)
    return f"audio/{name}"


SENTS = {"nv1": "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death.",
         "nv2": "The pure devotee always remembers Krishna and never forgets Him for a single moment."}
SEEDS = [42, 123, 7]
# (rid, label, cls, source-dir, stem)
REFDEF = [("c0081d", "0081 - WINNER (most him-typical, has a shout)", "win", "dn", "ggs_l1_0081"),
          ("c0086d", "0086 - calm-modal alternative", "calm", "dn", "ggs_l1_0086"),
          ("c0112n", "0112 - old baseline (atypical)", "old", "noisy", "ggs_l1_0112")]

refblock = []
for rid, label, cls, src, stem in REFDEF:
    base = CLIPS_DN if src == "dn" else CLIPS
    refblock.append(dict(label=label, cls=cls, url=put(base / f"{stem}.wav", f"ref_{stem}_{src}.wav"),
                         cos=round(cos(embed(base / f"{stem}.wav"), ref), 3)))

anchor = dict(url=put(CLIPS / "ggs_l1_0063.wav", "real_ggs_l1_0063.wav"),
              cos=round(cos(embed(CLIPS / "ggs_l1_0063.wav"), ref), 3))

sents = []
for sid, txt in SENTS.items():
    refs_out = []
    for rid, label, cls, src, stem in REFDEF:
        seeds = []
        for s in SEEDS:
            f = SWEEP / f"r{rid}__lora150__{sid}__s{s}.wav"
            if f.exists():
                seeds.append(dict(seed=s, url=put(f, f"{rid}__{sid}__s{s}.wav"), cos=round(cos(embed(f), ref), 3)))
        mean = round(float(np.mean([x["cos"] for x in seeds])), 3) if seeds else None
        refs_out.append(dict(label=label, cls=cls, mean=mean, seeds=seeds))
    sents.append(dict(sid=sid, text=txt, refs=refs_out))

payload = dict(ceiling=ceil, refs=refblock, anchor=anchor, sents=sents)
print(f"ceiling {ceil} | refs {[r['cos'] for r in refblock]} | anchor {anchor['cos']}")
(OUT / "index.html").write_text(TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
print("wrote", OUT / "index.html")
