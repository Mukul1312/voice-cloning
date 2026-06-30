"""
build_clone_compare.py — "clone vs. him, by axis".

Runs the SAME pitch/formant analysis (from build_voice_anatomy) on (a) a pool of GGS's REAL clips
and (b) each LoRA checkpoint's clone samples + the base model, then overlays them so you can SEE
which axis is off and by how much:
  - ACCENT/TIMBRE -> his vowel-space cloud vs the clone's (do the clone's vowels sit where his do?)
  - CADENCE       -> his F0 distribution vs the clone's (is the pitch range/movement like his?)
A scorecard ranks the checkpoints by closeness to his real voice (the interpretable cousin of an
ECAPA speaker-similarity score).

Output -> voice_lab/clone_vs_real/  (servable: not under data/)
Usage  -> .venv/Scripts/python.exe scripts/build_clone_compare.py
"""
import importlib.util
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "out" / "lecture1" / "final"
CLIPS = FINAL / "clips"
SAMPLES = ROOT / "data" / "out" / "lecture1" / "samples"
OUT = ROOT / "voice_lab" / "clone_vs_real"
AUD = OUT / "audio"

# import analyze() etc. from the anatomy builder (no module-level side effects beyond imports)
_spec = importlib.util.spec_from_file_location("bva", ROOT / "scripts" / "build_voice_anatomy.py")
bva = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bva)

N_REAL = 18           # his real clips to pool for the reference distribution
MAX_DOTS = 450        # cap vowel dots per group (render budget)
RNG = random.Random(7)

# clone variants -> their sample wavs.
#   plain   = voice baked into the model, no reference prompt
#   +prompt = generated with his real clip (ggs_l1_0112) as a voice prompt (the timbre-copy path)
VARIANTS = [
    ("base", "Base (no LoRA)", ["s150_plain1_lora_disabled.wav", "s150_plain2_lora_disabled.wav"]),
    ("150", "LoRA step 150", ["s150_plain1_with_lora.wav", "s150_plain2_with_lora.wav"]),
    ("250", "LoRA step 250", ["s250_plain1_with_lora.wav", "s250_plain2_with_lora.wav"]),
    ("350", "LoRA step 350", ["s350_plain1_with_lora.wav", "s350_plain2_with_lora.wav"]),
    ("500", "LoRA step 500", ["s500_plain1_with_lora.wav", "s500_plain2_with_lora.wav"]),
    ("basep", "Base + prompt", ["s150_clone1_lora_disabled.wav"]),
    ("150p", "step 150 + prompt", ["s150_clone1_with_lora.wav"]),
    ("250p", "step 250 + prompt", ["s250_clone1_with_lora.wav"]),
    ("350p", "step 350 + prompt", ["s350_clone1_with_lora.wav"]),
    ("500p", "step 500 + prompt", ["s500_clone1_with_lora.wav"]),
    # NEW: fresh same-config generations isolating the denoised reference (seed 42)
    ("250dnp", "step 250 + DENOISED prompt", ["s250_dnp_T1.wav", "s250_dnp_T2.wav"]),
    ("250np2", "step 250 + raw prompt", ["s250_np_T1.wav"]),
    ("basednp", "base + DENOISED prompt", ["base_dnp_T1.wav"]),
]


def pool(wavs):
    f0, vow = [], []
    for w in wavs:
        if not Path(w).exists():
            print(f"  (missing {Path(w).name})"); continue
        d, _, _ = bva.analyze(w)
        for i in range(d["n"]):
            if d["f0"][i] is not None:
                f0.append(float(d["f0"][i]))
            if d["F1"][i] is not None and d["F2"][i] is not None:
                vow.append([int(d["F1"][i]), int(d["F2"][i])])
    return f0, vow


def pool_spectral(wavs):
    """Long-Term Average Spectrum (LTAS, 0-8 kHz) — a speaker-level TIMBRE fingerprint (the average
    spectral shape, independent of which words are spoken)."""
    NB = 80
    grid = np.linspace(0, 8000, NB)
    acc = np.zeros(NB); cnt = 0
    for w in wavs:
        if not Path(w).exists():
            continue
        y, sr = librosa.load(str(w), sr=None)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        Sdb = librosa.amplitude_to_db(S, ref=np.max)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        keep = freqs <= 8000
        acc += np.interp(grid, freqs[keep], Sdb[keep, :].mean(axis=1)); cnt += 1
    return (acc / cnt).round(2).tolist() if cnt else None


def timbre_dist(a, b):
    """1 - correlation of two LTAS curves (lower = more similar timbre)."""
    if not a or not b:
        return None
    return round(float(1 - np.corrcoef(np.array(a), np.array(b))[0, 1]), 3)


def stats(f0, vow):
    f0 = np.array(f0) if f0 else np.array([0.0])
    F1 = np.array([v[0] for v in vow]) if vow else np.array([0])
    F2 = np.array([v[1] for v in vow]) if vow else np.array([0])
    return dict(f0_med=round(float(np.median(f0)), 1),
                f0_p10=round(float(np.percentile(f0, 10)), 0),
                f0_p90=round(float(np.percentile(f0, 90)), 0),
                cF1=int(round(float(np.median(F1)))), cF2=int(round(float(np.median(F2)))),
                f2_spread=int(round(float(np.std(F2)))))


def f0_hist(f0):
    bins = list(range(70, 261, 10))
    h, _ = np.histogram(f0, bins=bins)
    h = (h / h.max()).round(3).tolist() if h.max() else h.tolist()
    return dict(lo=70, hi=260, step=10, counts=h)


def thin(vow):
    return RNG.sample(vow, MAX_DOTS) if len(vow) > MAX_DOTS else vow


def pick_real():
    rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = set((ROOT / "data" / "english_words.txt").read_text(encoding="utf-8").split())
    tok = re.compile(r"[a-z']+")
    def ef(t):
        w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
    cand = [r for r in rows if 5 <= r["duration"] <= 12 and "[" not in r["text"] and ef(r["text"]) >= 0.85]
    cand.sort(key=lambda r: -r["duration"])
    return cand[:N_REAL]


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    if OUT.exists():
        shutil.rmtree(OUT)
    AUD.mkdir(parents=True, exist_ok=True)

    # --- his real reference ---
    real_rows = pick_real()
    real_wavs = [CLIPS / Path(r["audio"]).name for r in real_rows]
    print(f"pooling {len(real_wavs)} real clips for reference...")
    rf0, rvow = pool(real_wavs)
    ref = dict(id="real", label="GGS — real voice", f0hist=f0_hist(rf0), vowels=thin(rvow), stats=stats(rf0, rvow))
    ref["ltas"] = pool_spectral(real_wavs)
    shutil.copyfile(real_wavs[0], AUD / "real.wav")  # a listen sample
    ref["wav"] = "audio/real.wav"
    print(f"  real: {ref['stats']}")

    # --- clone variants ---
    variants = []
    for vid, label, files in VARIANTS:
        wavs = [SAMPLES / f for f in files]
        print(f"pooling {label}...")
        f0, vow = pool(wavs)
        if not f0:
            print(f"  (no data for {label}, skipping)"); continue
        first = next((w for w in wavs if w.exists()), None)
        wname = f"{vid}.wav"
        if first: shutil.copyfile(first, AUD / wname)
        v = dict(id=vid, label=label, f0hist=f0_hist(f0), vowels=thin(vow), stats=stats(f0, vow),
                 wav=f"audio/{wname}")
        # distance to his real voice (interpretable similarity): vowel centroid + vowel SPREAD
        # (his vowels are centralized; a clone with peripheral/spread vowels is "Anglicising") + pitch
        d_vow = float(np.hypot((v["stats"]["cF1"] - ref["stats"]["cF1"]) / 100.0,
                               (v["stats"]["cF2"] - ref["stats"]["cF2"]) / 250.0))
        d_spread = abs(v["stats"]["f2_spread"] - ref["stats"]["f2_spread"]) / 150.0
        d_f0 = abs(v["stats"]["f0_med"] - ref["stats"]["f0_med"]) / 20.0
        v["dist"] = round(d_vow + d_spread + d_f0, 3)
        v["d_vow"] = round(d_vow, 3); v["d_spread"] = round(d_spread, 3); v["d_f0"] = round(d_f0, 3)
        v["ltas"] = pool_spectral(wavs)
        v["timbre"] = timbre_dist(ref["ltas"], v["ltas"])
        variants.append(v)
        print(f"  {label}: f0={v['stats']['f0_med']} spread={v['stats']['f2_spread']} "
              f"cad+acc_dist={v['dist']} timbre={v['timbre']}")

    data = dict(ref=ref, variants=variants)
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT / 'index.html'}  ({len(variants)} clone variants vs real)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clone vs GGS — by axis</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;
        --real:#46c08a;--clone:#ff9d4a;--accent:#7c9cff;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:22px 18px 80px}
  h1{font-size:21px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 16px}
  .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 14px}
  .vbtn{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--mut);border-radius:8px;padding:6px 12px;font:inherit;font-weight:700}
  .vbtn.on{background:var(--clone);color:#0f1115;border-color:var(--clone)}
  .legend{font-size:13px;color:var(--mut)} .dot{display:inline-block;width:9px;height:9px;border-radius:50%;vertical-align:middle;margin:0 4px 0 10px}
  .panels{display:flex;gap:14px;flex-wrap:wrap}
  .panel{flex:1 1 360px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
  .plabel{font-size:13px;color:var(--mut);margin:0 0 6px}
  canvas{display:block;width:100%}
  button.play{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--ink);border-radius:7px;padding:5px 10px;font:inherit;font-size:12px}
  #score{background:var(--card);border:1px solid var(--accent);border-radius:12px;padding:12px 14px;margin:16px 0 0}
  table{width:100%;border-collapse:collapse;margin-top:8px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);font-size:13.5px}
  .win{color:var(--real);font-weight:700} .pill{font-size:11px;padding:1px 7px;border-radius:20px;background:#2a2e38}
  details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:0 0 16px}
  summary{cursor:pointer;font-weight:600} details ul{margin:8px 0 0;padding-left:18px;color:var(--mut)}
</style></head><body><div class="wrap">
  <h1>🆚 Clone vs. GGS — read it by axis</h1>
  <p class="sub">His <b style="color:var(--real)">real voice</b> (fixed reference) vs. a chosen <b style="color:var(--clone)">clone</b>. Same F0/formant analysis on both — so the gaps you see are real, measurable differences.</p>

  <details open>
    <summary>How to read this</summary>
    <ul>
      <li><b>Vowel map</b> — his real vowel cloud vs the clone's. <b>Accent/timbre:</b> if the clone's orange cloud drifts toward the grey standard-English vowels (more spread, more peripheral) while his green cloud stays central, the clone is "Anglicising" his vowels. The ✕ marks are the cloud centres.</li>
      <li><b>Pitch (F0)</b> — his pitch distribution vs the clone's. <b>Cadence:</b> a clone that's flatter / narrower than his spread is under-expressive (often overfit).</li>
      <li><b>Scorecard</b> — distance of each checkpoint to his real voice (lower = closer). The interpretable cousin of a speaker-similarity score: it tells you <i>which</i> checkpoint and <i>why</i>.</li>
    </ul>
  </details>

  <div class="row">
    <span style="color:var(--mut);font-size:13px">Compare clone:</span>
    <span id="vbtns"></span>
    <span class="legend"><span class="dot" style="background:var(--real)"></span>real <span class="dot" style="background:var(--clone)"></span>clone</span>
    <span style="flex:1"></span>
    <button class="play" id="pReal">▶ his voice</button>
    <button class="play" id="pClone">▶ clone</button>
  </div>

  <div class="panels">
    <div class="panel"><div class="plabel">🗺️ Vowel space — F1 (open↓) × F2 (front←) · accent</div><canvas id="vowel" height="280"></canvas></div>
    <div class="panel"><div class="plabel">🎵 Pitch (F0) distribution · cadence</div><canvas id="pitch" height="280"></canvas></div>
    <div class="panel"><div class="plabel">🎨 Average spectrum (LTAS, 0–8 kHz) · timbre</div><canvas id="ltas" height="280"></canvas></div>
  </div>

  <div id="score"></div>
</div>
<audio id="aReal"></audio><audio id="aClone"></audio>
<script>
const DATA = /*__DATA__*/;
const VOWELS=[ {s:"i",F1:270,F2:2290},{s:"ɪ",F1:390,F2:1990},{s:"ɛ",F1:530,F2:1840},{s:"æ",F1:660,F2:1720},
 {s:"ɑ",F1:730,F2:1090},{s:"ɔ",F1:570,F2:840},{s:"ʊ",F1:440,F2:1020},{s:"u",F1:300,F2:870},{s:"ʌ",F1:640,F2:1190},{s:"ɝ",F1:490,F2:1350}];
const $=s=>document.querySelector(s), css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let cur=DATA.variants.find(v=>v.id!=="base")||DATA.variants[0];

const VP={F2MIN:600,F2MAX:2600,F1MIN:200,F1MAX:900,pad:34};
function vX(f2,W){return VP.pad+(VP.F2MAX-f2)/(VP.F2MAX-VP.F2MIN)*(W-VP.pad*2);}
function vY(f1,H){return VP.pad+(f1-VP.F1MIN)/(VP.F1MAX-VP.F1MIN)*(H-VP.pad*2);}
function cloud(g,W,H,pts,color){g.fillStyle=color;for(const[f1,f2]of pts){g.beginPath();g.arc(vX(f2,W),vY(f1,H),2,0,7);g.fill();}}
function centroid(g,W,H,f1,f2,color){const x=vX(f2,W),y=vY(f1,H);g.strokeStyle=color;g.lineWidth=3;
  g.beginPath();g.moveTo(x-7,y-7);g.lineTo(x+7,y+7);g.moveTo(x+7,y-7);g.lineTo(x-7,y+7);g.stroke();g.lineWidth=1;}
function drawVowel(){const c=$("#vowel");c.width=c.parentElement.clientWidth-24;const g=c.getContext("2d"),W=c.width,H=c.height;
  g.clearRect(0,0,W,H);g.strokeStyle="#21252e";g.strokeRect(VP.pad,VP.pad,W-VP.pad*2,H-VP.pad*2);
  g.fillStyle="#6b7280";g.font="10px system-ui";g.fillText("← F2 front",VP.pad,14);g.fillText("back →",W-50,14);
  g.font="11px system-ui";for(const v of VOWELS){const x=vX(v.F2,W),y=vY(v.F1,H);g.strokeStyle="#5b6472";g.beginPath();g.arc(x,y,3,0,7);g.stroke();g.fillStyle="#7b8494";g.fillText(v.s,x+5,y+3);}
  cloud(g,W,H,DATA.ref.vowels,"rgba(70,192,138,.45)");
  cloud(g,W,H,cur.vowels,"rgba(255,157,74,.5)");
  centroid(g,W,H,DATA.ref.stats.cF1,DATA.ref.stats.cF2,css("--real"));
  centroid(g,W,H,cur.stats.cF1,cur.stats.cF2,css("--clone"));}
function drawHist(g,W,H,hist,color,base){const n=hist.counts.length;g.fillStyle=color;
  for(let i=0;i<n;i++){const x=VP.pad+i/n*(W-VP.pad*2),w=(W-VP.pad*2)/n-2,h=hist.counts[i]*(H-VP.pad*2);
    g.fillRect(x+(base?w*0.0:0),H-VP.pad-h,w,h);}}
function drawPitch(){const c=$("#pitch");c.width=c.parentElement.clientWidth-24;const g=c.getContext("2d"),W=c.width,H=c.height;
  g.clearRect(0,0,W,H);const hr=DATA.ref.f0hist;
  g.fillStyle="#6b7280";g.font="10px system-ui";
  for(let f=100;f<=250;f+=50){const i=(f-hr.lo)/(hr.hi-hr.lo);const x=VP.pad+i*(W-VP.pad*2);g.strokeStyle="#21252e";g.beginPath();g.moveTo(x,VP.pad);g.lineTo(x,H-VP.pad);g.stroke();g.fillText(f+"Hz",x-9,H-VP.pad+12);}
  drawHist(g,W,H,DATA.ref.f0hist,"rgba(70,192,138,.55)",false);
  drawHist(g,W,H,cur.f0hist,"rgba(255,157,74,.55)",true);
  g.fillStyle=css("--real");g.fillText("real med "+DATA.ref.stats.f0_med+"Hz",VP.pad+4,VP.pad+12);
  g.fillStyle=css("--clone");g.fillText("clone med "+cur.stats.f0_med+"Hz",VP.pad+4,VP.pad+26);}
function drawLTAS(){const c=$("#ltas");c.width=c.parentElement.clientWidth-24;const g=c.getContext("2d"),W=c.width,H=c.height,pad=30;
  g.clearRect(0,0,W,H);g.fillStyle="#6b7280";g.font="10px system-ui";
  for(let f=2000;f<8000;f+=2000){const x=pad+f/8000*(W-pad*2);g.strokeStyle="#21252e";g.beginPath();g.moveTo(x,pad);g.lineTo(x,H-pad);g.stroke();g.fillText((f/1000)+"k",x-6,H-pad+12);}
  function curve(arr,color){if(!arr||!arr.length)return;const lo=Math.min(...arr),hi=Math.max(...arr);
    g.strokeStyle=color;g.lineWidth=2;g.beginPath();
    for(let i=0;i<arr.length;i++){const x=pad+i/(arr.length-1)*(W-pad*2),v=(arr[i]-lo)/(hi-lo+1e-9),y=H-pad-v*(H-pad*2);i?g.lineTo(x,y):g.moveTo(x,y);}g.stroke();g.lineWidth=1;}
  curve(DATA.ref.ltas,css("--real"));curve(cur.ltas,css("--clone"));
  g.fillStyle=css("--clone");g.fillText("timbre dist "+(cur.timbre==null?"—":cur.timbre)+" (lower=closer)",pad+4,pad+12);}

function buildBtns(){const box=$("#vbtns");box.innerHTML="";
  DATA.variants.forEach(v=>{const b=document.createElement("button");b.className="vbtn"+(v.id===cur.id?" on":"");
    b.textContent=v.label.replace("LoRA ","");b.onclick=()=>{cur=v;buildBtns();redraw();};box.appendChild(b);});}
function redraw(){drawVowel();drawPitch();drawLTAS();
  $("#aClone").src=cur.wav;}
function drawScore(){const best=DATA.variants.filter(v=>v.id!=="base").slice().sort((a,b)=>a.dist-b.dist)[0];
  let h=`<b>📊 Scorecard — distance to his real voice (lower = closer)</b>`;
  if(best)h+=`<p>Closest checkpoint by anatomy: <span class="win">${best.label}</span> (dist ${best.dist}). `+
    `Vowel-space gap ${best.d_vow}, pitch gap ${best.d_f0}.</p>`;
  h+=`<table><tr><th>Variant</th><th>Cad+Acc dist</th><th>Pitch gap</th><th>Spread gap</th><th>Timbre dist</th><th>F0 med</th></tr>`;
  DATA.variants.slice().sort((a,b)=>a.dist-b.dist).forEach(v=>{
    const pr=v.id.endsWith("p");
    h+=`<tr><td>${v.label} ${v.id==="base"?'<span class="pill">control</span>':pr?'<span class="pill" style="background:#3a2e1a">prompt</span>':''}</td>`+
      `<td class="${best&&v.id===best.id?'win':''}">${v.dist}</td><td>${v.d_f0}</td><td>${v.d_spread}</td>`+
      `<td>${v.timbre==null?'—':v.timbre}</td><td>${v.stats.f0_med}</td></tr>`;});
  h+=`</table><p style="color:var(--mut);font-size:12.5px">His real: F0 med ${DATA.ref.stats.f0_med}Hz, vowel centre ${DATA.ref.stats.cF1}, ${DATA.ref.stats.cF2}, F2 spread ${DATA.ref.stats.f2_spread}.</p>`;
  $("#score").innerHTML=h;}

$("#pReal").onclick=()=>{$("#aReal").src=DATA.ref.wav;$("#aReal").play();};
$("#pClone").onclick=()=>{$("#aClone").play();};
window.addEventListener("resize",redraw);
buildBtns();redraw();drawScore();
</script></body></html>
"""

if __name__ == "__main__":
    main()
