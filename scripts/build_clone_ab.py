"""
build_clone_ab.py — EAR-CHECK: original-data clone  vs  denoised-data clone, same sentence + seed.

ECAPA said denoising slightly hurt (0.743 -> 0.704/0.727). But ear and ECAPA diverged on the denoise
gate, so judge it directly: for each checkpoint x sentence, hear the two clones side by side across all
seeds, with his REAL voice as the anchor and the per-clip cosine shown so you can see where the number
and the ear disagree.

Output -> voice_lab/clone_ab/  (servable: not under data/)
Run    -> .venv/Scripts/python.exe scripts/build_clone_ab.py
"""
import json
import re
import sys
import shutil
from collections import defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data/out/lecture1/final"
CLIPS = FINAL / "clips"
CLIPS_DN = FINAL / "clips_dn"
EVAL_ORIG = ROOT / "data/out/lecture1/eval"
EVAL_DN = ROOT / "data/out/lecture1/eval_dn"
OUT = ROOT / "voice_lab" / "clone_ab"
DEV = "cpu"

SENTS = {
    "nm":  {"seen": True,  "text": "Namabhasa means offences are not completely gone. If offences will completely go, then pure name will rise"},
    "nv1": {"seen": False, "text": "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death."},
    "nv2": {"seen": False, "text": "The pure devotee always remembers Krishna and never forgets Him for a single moment."},
}

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
    return F.normalize(clf.encode_batch(w.to(DEV)).squeeze(0).squeeze(0), dim=0)


def cos(a, b):
    return float(torch.dot(a, b))


# --- centroids (same selection as the scorer) ---
rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCL = {"ggs_l1_0112", "ggs_l1_0114"}
sel = [Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
ref_names = sel[:20]
ref_noisy = F.normalize(torch.stack([embed(CLIPS / n) for n in ref_names]).mean(0), dim=0)
ref_dn = F.normalize(torch.stack([embed(CLIPS_DN / n) for n in ref_names]).mean(0), dim=0)
ceil_noisy = float(np.mean([cos(embed(CLIPS / n), ref_noisy) for n in sel[20:30]]))

pat = re.compile(r"c(?P<ckpt>\w+?)__(?P<sent>\w+?)__s(?P<seed>\d+)\.wav")


def main():
    A = OUT / "audio"
    for sub in ("orig", "dn", "real"):
        (A / sub).mkdir(parents=True, exist_ok=True)

    # his real anchors: the real Namabhasa (parallel to clone 'nm') + two English clips
    real = []
    anchors = [("ggs_l1_0112.wav", "real Namabhasa (same words as nm)")] + [(n, "real English") for n in sel[20:22]]
    for name, lab in anchors:
        if not (CLIPS / name).exists():
            continue
        shutil.copyfile(CLIPS / name, A / "real" / name)
        real.append(dict(wav=f"audio/real/{name}", label=lab, cos=round(cos(embed(CLIPS / name), ref_noisy), 3)))

    rows_out = []
    for p in sorted(EVAL_ORIG.glob("*.wav")):
        m = pat.match(p.name)
        if not m:
            continue
        dn = EVAL_DN / p.name
        if not dn.exists():
            continue
        shutil.copyfile(p, A / "orig" / p.name)
        shutil.copyfile(dn, A / "dn" / p.name)
        e_dn = embed(dn)
        rows_out.append(dict(
            ckpt=m["ckpt"], sent=m["sent"], seed=int(m["seed"]),
            orig=f"audio/orig/{p.name}", dn=f"audio/dn/{p.name}",
            co=round(cos(embed(p), ref_noisy), 3),
            cdn=round(cos(e_dn, ref_noisy), 3),
            cdd=round(cos(e_dn, ref_dn), 3),
        ))

    rows_out.sort(key=lambda r: (int(r["ckpt"]), r["sent"], r["seed"]))
    data = dict(ceil=round(ceil_noisy, 3), sents=SENTS, real=real, rows=rows_out,
                verdict=dict(orig_best=0.743, dn_best_noisy=0.704, dn_best_dn=0.727))
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT/'index.html'}  ({len(rows_out)} A/B pairs)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clone A/B — original-data vs denoised-data</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;--real:#7c9cff;--orig:#46c08a;--dn:#ff9d4a;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:20px 16px 80px}
  h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 14px;max-width:780px}
  .verdict{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:0 0 14px;font-size:13.5px;color:var(--mut)}
  .verdict b{color:var(--ink)}
  .ref{background:var(--card);border:1px solid var(--real);border-radius:12px;padding:10px 14px;margin:0 0 14px}
  .ref .t{color:var(--real);font-weight:700;margin:0 0 6px}
  .rrow{display:flex;gap:10px;align-items:center;margin:3px 0;font-size:12.5px;color:var(--mut)} .rrow audio{height:32px;flex:1;max-width:340px}
  .ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:0 0 6px}
  .grp{font-size:12px;color:var(--mut)} .grp button{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--mut);border-radius:7px;padding:4px 10px;font:inherit;font-weight:700;margin-left:4px}
  .grp button.on{background:var(--ink);color:#0f1115;border-color:var(--ink)}
  .stext{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:8px 0 10px;font-size:13px}
  .stext .seen{font-size:11px;padding:1px 7px;border-radius:20px;background:#3a2e1a;color:#ffb84a;margin-left:6px}
  .stext .uns{font-size:11px;padding:1px 7px;border-radius:20px;background:#16302a;color:#46c08a;margin-left:6px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin:0 0 8px}
  .ctop{font-size:12px;color:var(--mut);margin:0 0 4px;font-family:ui-monospace,monospace}
  .cols{display:flex;gap:12px;flex-wrap:wrap}
  .col{flex:1 1 320px}
  .lab{font-size:12px;margin:0 0 3px} .lab.o{color:var(--orig)} .lab.d{color:var(--dn)}
  audio{width:100%;height:34px}
</style></head><body><div class="wrap">
  <h1>🎧 Clone A/B — original-data vs denoised-data clone</h1>
  <p class="sub">Same sentence, same seed — only the <b>training data</b> differs. Judge by ear whether the <b style="color:var(--dn)">denoised-data</b> clone really sounds worse than the <b style="color:var(--orig)">original-data</b> one, or if ECAPA is being harsh. Anchor everything to <b style="color:var(--real)">his real voice</b>.</p>
  <div class="verdict" id="verdict"></div>
  <div class="ref" id="ref"></div>
  <div class="ctl">
    <div class="grp">checkpoint:<span id="ckb"></span></div>
    <div class="grp">sentence:<span id="snb"></span></div>
  </div>
  <div class="stext" id="stext"></div>
  <div id="list"></div>
</div>
<script>
const DATA = /*__DATA__*/;
const $=s=>document.querySelector(s);
const ckpts=[...new Set(DATA.rows.map(r=>r.ckpt))].sort((a,b)=>a-b);
const sents=["nm","nv1","nv2"].filter(s=>DATA.rows.some(r=>r.sent===s));
let curCk=ckpts.includes("150")?"150":ckpts[0], curSn=sents.includes("nv1")?"nv1":sents[0];
$("#verdict").innerHTML=`<b>ECAPA verdict (best unseen):</b> original-data <b style="color:var(--orig)">${DATA.verdict.orig_best}</b> · denoised-data <b style="color:var(--dn)">${DATA.verdict.dn_best_noisy}</b> (vs noisy) / <b style="color:var(--dn)">${DATA.verdict.dn_best_dn}</b> (confound-corrected) · his ceiling ${DATA.ceil}. Your ears decide if that gap is audible.`;
$("#ref").innerHTML=`<div class="t">🔵 his real voice (anchor)</div>`+DATA.real.map(r=>
  `<div class="rrow"><span>${r.label} · cos ${r.cos}</span><audio controls preload="none" src="${r.wav}"></audio></div>`).join("");
function btns(box,arr,cur,cb){box.innerHTML="";arr.forEach(v=>{const b=document.createElement("button");
  b.textContent=v;b.className=v===cur?"on":"";b.onclick=()=>cb(v);box.appendChild(b);});}
function render(){
  btns($("#ckb"),ckpts,curCk,v=>{curCk=v;render();});
  btns($("#snb"),sents,curSn,v=>{curSn=v;render();});
  const meta=DATA.sents[curSn];
  $("#stext").innerHTML=`“${meta.text}”`+(meta.seen?`<span class="seen">SEEN (training)</span>`:`<span class="uns">UNSEEN</span>`);
  const rows=DATA.rows.filter(r=>r.ckpt===curCk&&r.sent===curSn).sort((a,b)=>a.seed-b.seed);
  $("#list").innerHTML=rows.map(r=>`<div class="card">
    <div class="ctop">step ${r.ckpt} · ${r.sent} · seed ${r.seed}</div>
    <div class="cols">
      <div class="col"><div class="lab o">original-data clone · cos ${r.co}</div><audio controls preload="none" src="${r.orig}"></audio></div>
      <div class="col"><div class="lab d">denoised-data clone · cos ${r.cdn} <span style="color:var(--mut)">(dn-centroid ${r.cdd})</span></div><audio controls preload="none" src="${r.dn}"></audio></div>
    </div></div>`).join("");
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
