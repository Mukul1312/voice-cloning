"""
build_denoise_compare.py — REAL vs DENOISED, per clip, so we can judge whether Resemble-Enhance
preserved his identity or altered it. For all 140 clips it lays out, side by side:
  - audio A/B  (original  ↔  denoised)            -> hear it
  - spectrogram A/B  (0–8 kHz, dB)                -> SEE the noise floor removed (or harmonics smeared)
  - ECAPA cosine to his centroid (orig, dn, Δ)    -> measure the per-clip identity shift
Sorted worst-Δ first, with a Δ-distribution summary: a UNIFORM small drop = the noisy-centroid
confound (denoising is fine); a few BIG drops = those specific clips got mangled (drop them).

Output -> voice_lab/denoise_compare/  (servable: not under data/)
Run    -> .venv/Scripts/python.exe scripts/build_denoise_compare.py
"""
import json
import sys
import shutil
import types
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
# librosa lazy-loads submodules; resolve them BEFORE speechbrain (which registers a lazy `k2`
# stub that librosa's lazy_loader later trips on). Then stub k2 defensively.
import librosa
import librosa.feature
import librosa.core.audio  # noqa
_ = librosa.feature.melspectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
sys.modules.setdefault("k2", types.ModuleType("k2"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "out" / "lecture1" / "final"
ORIG = FINAL / "clips"
DN = FINAL / "clips_dn"
OUT = ROOT / "voice_lab" / "denoise_compare"
A_ORIG = OUT / "audio" / "orig"
A_DN = OUT / "audio" / "dn"
IMG = OUT / "img"
DEV = "cpu"

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


def spec_png(wav_path, out_png):
    if out_png.exists():
        return
    y, sr = librosa.load(str(wav_path), sr=16000)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=256)), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    keep = freqs <= 8000
    fig, ax = plt.subplots(figsize=(3.6, 1.5), dpi=80)
    ax.imshow(S[keep], origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(y) / sr, 0, 8], vmin=-80, vmax=0)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    for d in (A_ORIG, A_DN, IMG):
        d.mkdir(parents=True, exist_ok=True)

    # transcripts keyed by clip stem
    texts = {}
    mpath = FINAL / "train.jsonl"
    if mpath.exists():
        for l in mpath.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                texts[Path(r["audio"]).stem] = r.get("text", "")

    names = sorted(p.name for p in ORIG.glob("*.wav") if (DN / p.name).exists())
    print(f"{len(names)} clips present in both orig + denoised")

    # his centroid from ALL original clips (self-inclusion per clip is 1/N -> negligible)
    print("embedding originals (for centroid)...")
    e_orig = {n: embed(ORIG / n) for n in names}
    ref = F.normalize(torch.stack([e_orig[n] for n in names]).mean(0), dim=0)

    rows = []
    for i, n in enumerate(names):
        stem = Path(n).stem
        co = cos(e_orig[n], ref)
        cd = cos(embed(DN / n), ref)
        if not (A_ORIG / n).exists():
            shutil.copyfile(ORIG / n, A_ORIG / n)
        if not (A_DN / n).exists():
            shutil.copyfile(DN / n, A_DN / n)
        spec_png(ORIG / n, IMG / f"{stem}_o.png")
        spec_png(DN / n, IMG / f"{stem}_d.png")
        rows.append(dict(id=stem, text=texts.get(stem, ""), wav=n,
                         co=round(co, 3), cd=round(cd, 3), d=round(cd - co, 3)))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(names)}")

    rows.sort(key=lambda r: r["d"])  # worst (most negative) first
    co_m = float(np.mean([r["co"] for r in rows]))
    cd_m = float(np.mean([r["cd"] for r in rows]))
    deltas = np.array([r["d"] for r in rows])
    summary = dict(
        n=len(rows), co_mean=round(co_m, 3), cd_mean=round(cd_m, 3), d_mean=round(cd_m - co_m, 3),
        big=int((deltas < -0.05).sum()), mod=int(((deltas >= -0.05) & (deltas < -0.02)).sum()),
        ok=int((deltas >= -0.02).sum()), up=int((deltas >= 0).sum()),
    )
    print(f"summary: {summary}")

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT / 'index.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Denoise check — real vs denoised, per clip</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;
        --orig:#46c08a;--dn:#ff9d4a;--bad:#ff5d5d;--warn:#ffb84a;--good:#46c08a;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:20px 16px 80px}
  h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 14px;max-width:760px}
  .sum{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:0 0 14px;display:flex;gap:22px;flex-wrap:wrap;align-items:center}
  .stat b{font-size:19px} .stat span{color:var(--mut);font-size:12px;display:block}
  .bars{display:flex;gap:4px;flex:1;min-width:200px;align-items:center}
  .seg{height:16px;border-radius:3px} .key{font-size:11.5px;color:var(--mut)}
  .ctl{margin:0 0 10px;font-size:13px;color:var(--mut)} .ctl button{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--mut);border-radius:7px;padding:4px 10px;font:inherit;font-weight:600}
  .ctl button.on{background:var(--dn);color:#0f1115;border-color:var(--dn)}
  .clip{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:0 0 10px}
  .top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
  .cid{font-weight:700;font-family:ui-monospace,monospace;font-size:12.5px;color:var(--mut)}
  .txt{flex:1;min-width:200px;color:var(--ink)}
  .delta{font-weight:800;font-size:15px;padding:1px 8px;border-radius:20px}
  .cols{display:flex;gap:12px;margin-top:8px;flex-wrap:wrap}
  .col{flex:1 1 320px}
  .lab{font-size:12px;margin:0 0 4px} .lab.o{color:var(--orig)} .lab.d{color:var(--dn)}
  audio{width:100%;height:34px} img{width:100%;border-radius:6px;display:block;margin-top:5px;background:#000}
  .cos{font-size:12px;color:var(--mut)}
</style></head><body><div class="wrap">
  <h1>🔍 Denoise check — real vs denoised, per clip</h1>
  <p class="sub">For each clip: hear <b style="color:var(--orig)">original</b> vs <b style="color:var(--dn)">denoised</b>, see both spectrograms (the broadband haze that disappears is the noise we removed), and the per-clip <b>ECAPA cosine</b> to his voice. Sorted <b>worst-shift first</b>. The question: does denoised still sound like <i>him, cleaner</i> — or did his voice change?</p>
  <div class="sum" id="sum"></div>
  <div class="ctl">sort: <button data-s="d" class="on">worst Δ first</button> <button data-s="id">clip order</button> <button data-s="co">lowest original first</button></div>
  <div id="list"></div>
</div>
<script>
const DATA = /*__DATA__*/;
const $=s=>document.querySelector(s);
function dcolor(d){return d<=-0.05?"var(--bad)":d<=-0.02?"var(--warn)":d>=0?"var(--good)":"var(--mut)";}
function renderSum(){const s=DATA.summary;const tot=s.n;
  const seg=(n,c,t)=>`<div class="seg" style="flex:${n||0.001};background:${c}" title="${t}: ${n}"></div>`;
  $("#sum").innerHTML=
    `<div class="stat"><b style="color:var(--orig)">${s.co_mean}</b><span>orig mean cos</span></div>`+
    `<div class="stat"><b style="color:var(--dn)">${s.cd_mean}</b><span>denoised mean cos</span></div>`+
    `<div class="stat"><b style="color:${dcolor(s.d_mean)}">${s.d_mean}</b><span>mean Δ</span></div>`+
    `<div class="bars">${seg(s.big,'var(--bad)','Δ<-0.05')}${seg(s.mod,'var(--warn)','-0.05..-0.02')}${seg(s.ok-s.up,'var(--mut)','-0.02..0')}${seg(s.up,'var(--good)','Δ≥0')}</div>`+
    `<div class="key">${s.big} big · ${s.mod} mod · ${s.ok} small/up of ${tot}</div>`;}
function row(r){return `<div class="clip">
  <div class="top"><span class="cid">${r.id}</span><span class="txt">${r.text||""}</span>
    <span class="delta" style="background:${dcolor(r.d)};color:#0f1115">Δ ${r.d>=0?'+':''}${r.d}</span></div>
  <div class="cols">
    <div class="col"><div class="lab o">original · cos ${r.co}</div>
      <audio controls preload="none" src="audio/orig/${r.wav}"></audio><img loading="lazy" src="img/${r.id}_o.png"></div>
    <div class="col"><div class="lab d">denoised · cos ${r.cd}</div>
      <audio controls preload="none" src="audio/dn/${r.wav}"></audio><img loading="lazy" src="img/${r.id}_d.png"></div>
  </div></div>`;}
let sortKey="d";
function render(){const rows=DATA.rows.slice();
  if(sortKey==="d")rows.sort((a,b)=>a.d-b.d);
  else if(sortKey==="co")rows.sort((a,b)=>a.co-b.co);
  else rows.sort((a,b)=>a.id<b.id?-1:1);
  $("#list").innerHTML=rows.map(row).join("");}
document.querySelectorAll(".ctl button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll(".ctl button").forEach(x=>x.classList.remove("on"));
  b.classList.add("on");sortKey=b.dataset.s;render();});
renderSum();render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
