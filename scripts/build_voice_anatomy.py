"""
build_voice_anatomy.py — an interactive "hear it / see it / measure it" view of GGS's real voice,
for learning the three axes of a voice on his actual clips:
  - CADENCE  -> the F0 (pitch) contour over time          (Praat SHS pitch, de-doubled)
  - TIMBRE   -> the spectral envelope                       (wideband spectrogram + a live spectrum slice)
  - ACCENT   -> formant positions on vowels (F1/F2)         (formant overlay + an F1-F2 vowel map)

Precomputes the DSP (Praat-grade F0 + formants via parselmouth, a wideband spectrogram via librosa)
for a handful of his clean English clips and writes a SELF-CONTAINED HTML page (data embedded, works
on file://). The page plays/scrubs the clip with time-aligned panels, a live numeric readout, a
spectrum-slice inspector (timbre), an F1-F2 vowel map vs standard English (accent), and a
drag-to-loop / 0.5x mechanic for focused ear-training.

Usage:  .venv/Scripts/python.exe scripts/build_voice_anatomy.py
Open:   data/out/lecture1/anatomy/voice_anatomy.html
"""
import base64
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import parselmouth

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "out" / "lecture1" / "final"
CLIPS = FINAL / "clips"
OUT = ROOT / "data" / "out" / "lecture1" / "anatomy"

# --- analysis params (tuned for an adult male voice on old recordings) ---
DT = 0.01            # 10 ms grid for F0 + formants
PITCH_FLOOR = 70.0   # pitch DISPLAY range (Hz)
PITCH_CEIL = 250.0
SHS_CEIL = 300.0     # internal SHS tracker ceiling
MAX_FORMANT = 5000.0 # Praat male default ~5000 Hz
SPEC_FMAX = 5000.0   # display 0..5 kHz (where formants live)
DB_FLOOR = -70.0
N_WAVE = 1100        # waveform envelope columns
NF_SLICE = 110       # freq bins for the spectrum-slice inspector
DT_SLICE = 0.02      # time step for the slice matrix


def pick_clips():
    rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    eng = set((ROOT / "data" / "english_words.txt").read_text(encoding="utf-8").split())
    tok = re.compile(r"[a-z']+")
    def efrac(t):
        w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
    by_name = {Path(r["audio"]).stem: r for r in rows}
    chosen = []
    if "ggs_l1_0112" in by_name:
        chosen.append(by_name["ggs_l1_0112"])
    cand = [r for r in rows
            if 5.0 <= r["duration"] <= 12.0 and "[" not in r["text"]
            and "hmm" not in r["text"].lower() and efrac(r["text"]) >= 0.85
            and Path(r["audio"]).stem != "ggs_l1_0112"]
    cand.sort(key=lambda r: r["duration"])
    if cand:
        idxs = sorted(set(int(x) for x in np.linspace(0, len(cand) - 1, num=min(4, len(cand)))))
        chosen += [cand[i] for i in idxs]
    return chosen


def _stft_db(y, sr):
    n_fft = 512
    win = max(16, int(0.006 * sr))   # ~6 ms -> WIDEBAND (formants show as bands)
    hop = max(8, int(0.002 * sr))
    S = np.abs(librosa.stft(y, n_fft=n_fft, win_length=win, hop_length=hop, window="hann"))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    keep = freqs <= SPEC_FMAX
    return S_db[keep, :], float(freqs[keep][-1]), hop


def spectrogram_png(S_db, png_path):
    norm = np.clip((S_db - DB_FLOOR) / (0.0 - DB_FLOOR), 0, 1)
    plt.imsave(str(png_path), norm[::-1, :], cmap="magma", vmin=0, vmax=1)


def slice_matrix(S_db, fmax, hop, sr):
    """Downsampled magnitude matrix (uint8, base64) so the browser can draw a spectrum slice at the cursor."""
    fidx = np.linspace(0, S_db.shape[0] - 1, NF_SLICE).astype(int)
    M = S_db[fidx, :]
    frame_dt = hop / sr
    step = max(1, int(round(DT_SLICE / frame_dt)))
    M = M[:, ::step]
    nt = M.shape[1]
    norm = np.clip((M - DB_FLOOR) / (0.0 - DB_FLOOR), 0, 1)
    u8 = (norm * 255).astype(np.uint8)        # (NF_SLICE, nt)
    flat = np.ascontiguousarray(u8.T).tobytes()  # time-major: frame f -> NF_SLICE values
    return dict(nf=NF_SLICE, nt=nt, dt=round(step * frame_dt, 4), fmax=round(fmax, 1),
                data=base64.b64encode(flat).decode())


def _val(v):
    return float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float("nan")


def median3(a):
    b = a.copy()
    for i in range(1, len(a) - 1):
        w = [x for x in (a[i - 1], a[i], a[i + 1]) if not math.isnan(x)]
        if w:
            b[i] = float(np.median(w))
    return b


def robust_f0(snd, n):
    """Pitch robust to weak/missing fundamentals (old recordings):
    subharmonic summation -> octave-correct toward the SHS median -> 3-pt median smooth."""
    shs = snd.to_pitch_shs(time_step=DT, minimum_pitch=75, maximum_frequency_component=1250,
                           max_number_of_subharmonics=15, ceiling=SHS_CEIL)
    arr = np.array([_val(shs.get_value_at_time(i * DT)) for i in range(n)])
    voiced = arr[~np.isnan(arr)]
    if len(voiced) >= 5:
        M = float(np.median(voiced))
        for i in range(len(arr)):
            if not math.isnan(arr[i]):
                while arr[i] > 1.6 * M and arr[i] / 2 >= 75:
                    arr[i] /= 2
        arr = median3(arr)
    return arr


def find_clear_vowels(f0, F1, F2, dt, max_n=6):
    """Find steady vowel nuclei: runs of voiced frames where F1/F2 sit still (a sustained vowel),
    so the learner gets clean, loopable targets instead of fast-moving speech."""
    n = len(f0)
    def ok(i): return f0[i] is not None and F1[i] is not None and F2[i] is not None
    stable = [False] * n
    for i in range(n):
        win = [j for j in range(max(0, i - 3), min(n, i + 4)) if ok(j)]
        if ok(i) and len(win) >= 5 and np.std([F1[j] for j in win]) < 70 and np.std([F2[j] for j in win]) < 150:
            stable[i] = True
    runs, i = [], 0
    while i < n:
        if stable[i]:
            j = i
            while j < n and stable[j]:
                j += 1
            if (j - i) >= 8:  # >= 80 ms steady
                seg = [k for k in range(i, j) if ok(k)]
                runs.append(dict(
                    t0=round(i * dt, 3), t1=round(j * dt, 3), tc=round((i + j) / 2 * dt, 3),
                    F1=int(round(np.median([F1[k] for k in seg]))),
                    F2=int(round(np.median([F2[k] for k in seg]))),
                    f0=round(float(np.median([f0[k] for k in seg])), 1),
                    dur=j - i))
            i = j
        else:
            i += 1
    runs.sort(key=lambda r: -r["dur"])
    picked = []
    for r in runs:                       # spread out: skip near-duplicates
        if all(abs(r["tc"] - p["tc"]) > 0.3 for p in picked):
            picked.append(r)
        if len(picked) >= max_n:
            break
    picked.sort(key=lambda r: r["tc"])
    return picked


def analyze(wav):
    snd = parselmouth.Sound(str(wav))
    dur = float(snd.get_total_duration())
    formant = snd.to_formant_burg(time_step=DT, max_number_of_formants=5,
                                  maximum_formant=MAX_FORMANT, window_length=0.025,
                                  pre_emphasis_from=50.0)
    n = int(math.floor(dur / DT))
    times = [round(i * DT, 3) for i in range(n)]
    f0arr = robust_f0(snd, n)
    f0, F1, F2, F3 = [], [], [], []

    def fnum(k, t):
        try:
            v = formant.get_value_at_time(k, t)
        except Exception:
            v = None
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(v, 0)

    for i, t in enumerate(times):
        f = f0arr[i]
        voiced = not math.isnan(f)
        f0.append(round(float(f), 1) if voiced else None)
        if voiced:
            F1.append(fnum(1, t)); F2.append(fnum(2, t)); F3.append(fnum(3, t))
        else:
            F1.append(None); F2.append(None); F3.append(None)

    vowels = find_clear_vowels(f0, F1, F2, DT)
    y, sr = librosa.load(str(wav), sr=None)
    N = N_WAVE; L = len(y)
    env = [float(np.max(np.abs(y[i * L // N:(i + 1) * L // N])) if (i + 1) * L // N > i * L // N else 0.0)
           for i in range(N)]
    m = max(env) or 1.0
    env = [round(e / m, 4) for e in env]
    return dict(dur=round(dur, 3), sr=int(sr), dt=DT, n=n, vowels=vowels,
                f0=f0, F1=F1, F2=F2, F3=F3, wave=env), y, sr


def med(vals):
    v = [x for x in vals if x is not None]
    return round(float(np.median(v)), 1) if v else None


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    if OUT.exists():
        for p in OUT.glob("*"):
            if p.is_file(): p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = pick_clips()
    print(f"selected {len(rows)} clips")
    clips = []
    for r in rows:
        stem = Path(r["audio"]).stem
        wav = CLIPS / Path(r["audio"]).name
        data, y, sr = analyze(wav)
        S_db, fmax, hop = _stft_db(y, sr)
        spectrogram_png(S_db, OUT / f"{stem}_spec.png")
        sl = slice_matrix(S_db, fmax, hop, sr)
        shutil.copyfile(wav, OUT / f"{stem}.wav")
        clip = dict(id=stem, text=r["text"], wav=f"{stem}.wav", png=f"{stem}_spec.png",
                    fmax=round(fmax, 1), pitch_lo=PITCH_FLOOR, pitch_hi=PITCH_CEIL, slice=sl, **data)
        clips.append(clip)
        print(f"  {stem}: dur={data['dur']}s  medF0={med(data['f0'])}Hz  "
              f"medF1={med(data['F1'])}  medF2={med(data['F2'])}  medF3={med(data['F3'])}")

    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(clips, ensure_ascii=False))
    (OUT / "voice_anatomy.html").write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT / 'voice_anatomy.html'}  ({len(clips)} clips)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice anatomy — Gour Govinda Swami</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;
        --f1:#ff5d5d;--f2:#ffd24a;--f3:#5dd0ff;--pitch:#7CFC9B;--accent:#7c9cff;--sel:#7c9cff;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:22px 18px 80px}
  h1{font-size:21px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 16px}
  details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:0 0 16px}
  summary{cursor:pointer;font-weight:600}
  details ul{margin:8px 0 0;padding-left:18px;color:var(--mut)} details li{margin:3px 0}
  #clips{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}
  .cbtn{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--mut);
        border-radius:8px;padding:6px 12px;font:inherit;font-size:13px}
  .cbtn.on{background:var(--accent);color:#0f1115;border-color:var(--accent);font-weight:700}
  .text{background:#11141b;border:1px solid var(--line);border-radius:8px;padding:10px 12px;
        margin:0 0 12px;font-style:italic;color:#cfd4de}
  .controls{display:flex;align-items:center;gap:10px;margin:0 0 10px;flex-wrap:wrap}
  button.btn{cursor:pointer;border:1px solid var(--line);background:var(--card);color:var(--ink);
        border-radius:8px;padding:7px 13px;font:inherit;font-weight:600}
  #play{background:var(--accent);color:#0f1115;border-color:var(--accent)}
  .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
  .seg button{border:0;background:#11141b;color:var(--mut);padding:7px 11px;cursor:pointer;font:inherit}
  .seg button.on{background:var(--accent);color:#0f1115;font-weight:700}
  button.tog.on{background:#2a3350;border-color:var(--accent);color:#cfe0ff}
  #readout{font-variant-numeric:tabular-nums;font-size:13px;color:var(--mut);margin-left:auto}
  #readout b{color:var(--ink)}
  .chip{display:inline-block;padding:1px 6px;border-radius:5px;font-weight:700;font-size:12px;color:#0f1115}
  .stage{position:relative;background:var(--card);border:1px solid var(--line);border-radius:12px;
         padding:8px 8px 4px;cursor:crosshair;user-select:none}
  .panel{margin:0 0 6px}
  .plabel{font-size:12px;color:var(--mut);margin:2px 2px 3px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  .plabel .leg{font-weight:700}
  canvas{display:block;width:100%}
  .specwrap{position:relative;width:100%}
  .specwrap img{display:block;width:100%;height:220px;border-radius:4px}
  .specwrap canvas{position:absolute;left:0;top:0}
  #playhead{position:absolute;top:6px;bottom:4px;width:2px;background:#fff;opacity:.9;pointer-events:none;left:8px}
  #hover{position:absolute;top:6px;bottom:4px;width:0;border-left:1px dashed #9aa3b2;opacity:.7;pointer-events:none;left:8px;display:none}
  #sel{position:absolute;top:6px;bottom:4px;background:rgba(124,156,255,.16);border-left:1px solid var(--sel);
       border-right:1px solid var(--sel);pointer-events:none;left:8px;width:0;display:none}
  .inspectors{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
  .ipanel{flex:1 1 320px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 10px}
  .hint{color:var(--mut);font-size:12.5px;margin:10px 2px 0}
  .where{font-size:11px;color:#6b7280;font-weight:600;background:#0f1115;border:1px solid var(--line);border-radius:5px;padding:1px 6px}
  .guide p{margin:8px 0;padding:9px 11px;background:#11141b;border:1px solid var(--line);border-radius:8px;line-height:1.55}
  .guide .h{font-size:14px} .ctrls{color:var(--mut);font-size:13px}
  .vbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 14px;font-size:13px;color:var(--mut)}
  .vbtn{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--ink);border-radius:8px;padding:5px 11px;font:inherit;font-weight:700}
  .vbtn:hover{border-color:var(--accent);background:#1c2230}
</style></head>
<body><div class="wrap">
  <h1>🗣️ Voice anatomy — hear it, see it, measure it</h1>
  <p class="sub">Play, hover, or drag-to-loop a clip of his real voice. Top panels share a timeline; the two inspectors below show the cursor moment.</p>

  <details open>
    <summary>How to read this — the 3 axes of a voice (click to collapse)</summary>
    <p style="color:var(--mut);margin:10px 0 6px">One voice, shown three ways. Each panel isolates one thing your ear normally blends together — and the readout puts a <b>number</b> on it. A voice = <b>cadence</b> (its tune) + <b>timbre</b> (its colour) + <b>accent</b> (how it shapes sounds). They're independent: two people can share an accent and a cadence and still sound different — that leftover is timbre.</p>
    <div class="guide">
      <p><span class="h"><b style="color:var(--pitch)">① CADENCE — his melody &amp; pacing</b></span> &nbsp;<span class="where">top · Pitch panel · green line</span><br>
      <b>What:</b> how high or low the voice is, instant to instant. That's <b>pitch = F0</b> (fundamental frequency), in Hz — his vocal folds open/close ~130 times a second, so ~130 Hz. <b>Watch:</b> the green line <i>rises</i> on stressed / emphatic words and <i>falls</i> at the end of a phrase. That rise-and-fall <i>shape</i> is exactly the "pace" you already hear — you're just seeing it now.</p>
      <p><span class="h"><b style="color:var(--accent)">② TIMBRE — his tone-colour</b></span> &nbsp;<span class="where">bottom-left · Spectrum slice</span><br>
      <b>What:</b> what makes him sound like <i>him</i>, separate from pitch or words. His vocal-tract shape (throat, mouth) <i>resonates</i> — boosting certain frequency bands called <b>formants</b>. The set of those boosts is his timbre. <b>Watch:</b> the curve's overall <i>shape</i>; the <b>bumps</b> are the formants (F1/F2/F3 are marked). As he moves between sounds the bumps slide — that sliding shape <i>is</i> timbre in motion. (The spectrogram just above is the same information drawn as a heat-map over time.)</p>
      <p><span class="h"><b style="color:var(--f2)">③ ACCENT — how he forms vowels</b></span> &nbsp;<span class="where">bottom-right · Vowel map</span><br>
      <b>What:</b> every vowel is (roughly) a single point fixed by its first two formants — <b>F1</b> ≈ how open the mouth is, <b>F2</b> ≈ how far forward the tongue is. <b style="color:#fff">●</b> = <i>his</i> current vowel; <span style="color:#8b93a3">○</span> = where a standard English speaker puts each vowel (heed, hid, head, had…). <b>Watch:</b> when his white dot lands <i>away</i> from the grey reference points, that gap is his accent — e.g. an "ah" sitting somewhere English doesn't put it.</p>
    </div>
    <p style="margin:8px 0 4px"><b>The readout</b> (the line just above the panels) shows live <b>t · F0 · F1 · F2 · F3</b> at the cursor, so every sound has a number attached.</p>
    <p class="ctrls" style="margin:4px 0 0"><b>Controls:</b> &nbsp;<b>Click</b> = move the playhead (it sticks) &nbsp;·&nbsp; <b>Hover</b> = dashed line, inspect any moment without moving the playhead &nbsp;·&nbsp; <b>Drag</b> across the waveform = select a region &nbsp;·&nbsp; <b>↻ Loop</b> + <b>0.5×</b> = repeat that region slowly with pitch kept (the ear-training drill) &nbsp;·&nbsp; <b>Space</b> = play / pause.</p>
  </details>

  <div id="clips"></div>
  <div class="text" id="cliptext"></div>
  <div class="controls">
    <button class="btn" id="play">▶ Play</button>
    <span class="seg" id="speed"><button class="on" data-s="1">1×</button><button data-s="0.5">0.5×</button><button data-s="0.25">0.25×</button></span>
    <button class="btn tog" id="loop">↻ Loop</button>
    <button class="btn" id="clearsel">⬚ Clear</button>
    <span id="readout"></span>
  </div>

  <div class="vbar" id="vbar"></div>

  <div class="stage" id="stage">
    <div class="panel"><div class="plabel">Waveform <span style="color:var(--mut)">— drag here to select a region</span></div><canvas id="wave" height="80"></canvas></div>
    <div class="panel"><div class="plabel"><span class="leg" style="color:var(--pitch)">● Pitch F0 (cadence)</span><span id="plab"></span></div><canvas id="pitch" height="150"></canvas></div>
    <div class="panel">
      <div class="plabel">Spectrogram + formants (timbre &amp; accent)
        <span class="leg" style="color:var(--f1)">● F1</span><span class="leg" style="color:var(--f2)">● F2</span><span class="leg" style="color:var(--f3)">● F3</span>
      </div>
      <div class="specwrap"><img id="specimg" alt="spectrogram"><canvas id="formants" height="220"></canvas></div>
    </div>
    <div id="sel"></div><div id="hover"></div><div id="playhead"></div>
  </div>

  <div class="inspectors">
    <div class="ipanel">
      <div class="plabel">🎨 Spectrum at cursor — the <b>shape</b> is his timbre; the bumps are formants</div>
      <canvas id="slice" height="190"></canvas>
    </div>
    <div class="ipanel">
      <div class="plabel">🗺️ Vowel map (F1–F2) — <span style="color:var(--accent)">●</span> his vowel vs <span style="color:#8b93a3">○</span> standard English</div>
      <canvas id="vowel" height="240"></canvas>
    </div>
  </div>
  <div class="hint">Tip: slowing to 0.5× keeps the pitch (time-stretch), so you hear articulation without the chipmunk effect.</div>
</div>
<audio id="audio" preload="auto"></audio>
<script>
const DATA = /*__DATA__*/;
// Peterson & Barney (1952) MALE average formants (Hz) — the standard English vowel reference.
const VOWELS=[
 {s:"i",w:"heed",F1:270,F2:2290},{s:"ɪ",w:"hid",F1:390,F2:1990},{s:"ɛ",w:"head",F1:530,F2:1840},
 {s:"æ",w:"had",F1:660,F2:1720},{s:"ɑ",w:"hod",F1:730,F2:1090},{s:"ɔ",w:"hawed",F1:570,F2:840},
 {s:"ʊ",w:"hood",F1:440,F2:1020},{s:"u",w:"who'd",F1:300,F2:870},{s:"ʌ",w:"hud",F1:640,F2:1190},
 {s:"ɝ",w:"heard",F1:490,F2:1350}];
const $=s=>document.querySelector(s);
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const stage=$("#stage"),audio=$("#audio"),readout=$("#readout");
const waveC=$("#wave"),pitchC=$("#pitch"),formC=$("#formants"),specImg=$("#specimg"),
      sliceC=$("#slice"),vowelC=$("#vowel"),playhead=$("#playhead"),hoverEl=$("#hover"),selEl=$("#sel");
let clip=null,W=0,sel=null,looping=false,playT=0,hoverT=null,down=false,downX=0,moved=false;

function b64u8(b){const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return u;}
function idxAt(t){return Math.max(0,Math.min(clip.n-1,Math.round(t/clip.dt)));}
function activeT(){return (!audio.paused)?audio.currentTime:(hoverT!=null?hoverT:playT);}
function nearestVowel(f1,f2){let best=VOWELS[0],bd=1e9;for(const v of VOWELS){const d=Math.hypot(f1-v.F1,(f2-v.F2)*0.4);if(d<bd){bd=d;best=v;}}return best;}
function setSpeed(s){audio.playbackRate=s;document.querySelectorAll("#speed button").forEach(x=>x.classList.toggle("on",+x.dataset.s===s));}

function buildClips(){const box=$("#clips");box.innerHTML="";
  DATA.forEach((c,i)=>{const b=document.createElement("button");b.className="cbtn"+(i===0?" on":"");
    b.textContent=c.id.replace("ggs_l1_","clip ");b.dataset.i=i;b.onclick=()=>select(i);box.appendChild(b);});}
function buildVowelBar(){const box=$("#vbar");
  box.innerHTML='<span>🎯 Study a clear vowel (one click = loop it at 0.25×):</span>';
  (clip.vowels||[]).forEach(v=>{const nv=nearestVowel(v.F1,v.F2);const b=document.createElement("button");
    b.className="vbtn";b.textContent=nv.s+" · "+nv.w;b.title="F1 "+v.F1+" / F2 "+v.F2+" Hz @ "+v.tc+"s";
    b.onclick=()=>studyVowel(v);box.appendChild(b);});}
function studyVowel(v){sel={a:v.t0,b:v.t1};looping=true;$("#loop").classList.add("on");setSpeed(0.25);
  playT=v.t0;audio.currentTime=v.t0;setCursorEls();refresh();audio.play();}
function select(i){clip=DATA[i];clip._sl=b64u8(clip.slice.data);
  document.querySelectorAll(".cbtn").forEach(b=>b.classList.toggle("on",+b.dataset.i===i));
  $("#cliptext").textContent="“"+clip.text+"”";
  $("#plab").textContent="  range "+clip.pitch_lo+"–"+clip.pitch_hi+" Hz";
  audio.src=clip.wav;audio.load();audio.pause();specImg.src=clip.png;
  sel=null;selEl.style.display="none";playT=0;hoverT=null;hoverEl.style.display="none";
  buildVowelBar();setPlay();layout();refresh();}
function layout(){W=stage.clientWidth-16;
  [waveC,pitchC,formC].forEach(c=>{c.width=W;c.style.width=W+"px";});
  specImg.style.width=W+"px";formC.height=220;formC.style.height="220px";
  sliceC.width=sliceC.parentElement.clientWidth-20;vowelC.width=vowelC.parentElement.clientWidth-20;
  drawWave();drawPitch();drawFormants();}
function drawWave(){const g=waveC.getContext("2d"),H=waveC.height;g.clearRect(0,0,W,H);
  g.strokeStyle="#2a2e38";g.beginPath();g.moveTo(0,H/2);g.lineTo(W,H/2);g.stroke();
  g.strokeStyle="#5b6472";g.beginPath();const N=clip.wave.length;
  for(let i=0;i<N;i++){const x=i/(N-1)*W,a=clip.wave[i]*(H/2-2);g.moveTo(x,H/2-a);g.lineTo(x,H/2+a);}g.stroke();}
function yP(f,H){return H*(1-(f-clip.pitch_lo)/(clip.pitch_hi-clip.pitch_lo));}
function drawPitch(){const g=pitchC.getContext("2d"),H=pitchC.height;g.clearRect(0,0,W,H);
  g.fillStyle="#6b7280";g.font="10px system-ui";g.strokeStyle="#21252e";
  for(let f=100;f<clip.pitch_hi;f+=50){const y=yP(f,H);g.beginPath();g.moveTo(0,y);g.lineTo(W,y);g.stroke();g.fillText(f+" Hz",3,y-2);}
  g.strokeStyle=css("--pitch");g.lineWidth=2;g.beginPath();let pen=false;
  for(let i=0;i<clip.n;i++){const f=clip.f0[i];if(f==null){pen=false;continue;}
    const x=i/(clip.n-1)*W,y=yP(f,H);if(!pen){g.moveTo(x,y);pen=true;}else g.lineTo(x,y);}g.stroke();g.lineWidth=1;}
function drawFormants(){const g=formC.getContext("2d"),H=formC.height,fmax=clip.fmax;g.clearRect(0,0,W,H);
  for(const[k,c]of[["F1","--f1"],["F2","--f2"],["F3","--f3"]]){g.fillStyle=css(c);
    for(let i=0;i<clip.n;i++){const f=clip[k][i];if(f==null)continue;const x=i/(clip.n-1)*W,y=H*(1-f/fmax);
      g.beginPath();g.arc(x,y,1.6,0,7);g.fill();}}}

function drawSlice(t){const g=sliceC.getContext("2d"),Wp=sliceC.width,H=sliceC.height,s=clip.slice,u=clip._sl;
  g.clearRect(0,0,Wp,H);const fr=Math.max(0,Math.min(s.nt-1,Math.round(t/s.dt)));
  g.fillStyle="#6b7280";g.font="10px system-ui";
  for(let f=1000;f<s.fmax;f+=1000){const x=f/s.fmax*Wp;g.strokeStyle="#21252e";g.beginPath();g.moveTo(x,0);g.lineTo(x,H-12);g.stroke();g.fillText((f/1000)+"k",x-5,H-1);}
  g.strokeStyle=css("--accent");g.lineWidth=2;g.beginPath();
  for(let f=0;f<s.nf;f++){let v=u[fr*s.nf+f];if(f>0&&f<s.nf-1)v=(u[fr*s.nf+f-1]+v+u[fr*s.nf+f+1])/3;
    const x=f/(s.nf-1)*Wp,y=(H-14)-v/255*(H-18);if(f===0)g.moveTo(x,y);else g.lineTo(x,y);}g.stroke();g.lineWidth=1;
  const i=idxAt(t);
  for(const[lab,fv,c]of[["F1",clip.F1[i],"--f1"],["F2",clip.F2[i],"--f2"],["F3",clip.F3[i],"--f3"]]){
    if(fv==null)continue;const x=fv/s.fmax*Wp;g.strokeStyle=css(c);g.beginPath();g.moveTo(x,0);g.lineTo(x,H-12);g.stroke();
    g.fillStyle=css(c);g.fillText(lab+" "+Math.round(fv),x+2,11);}}

const VP={F2MIN:600,F2MAX:2600,F1MIN:200,F1MAX:900,pad:30};
function vX(f2,Wp){return VP.pad+(VP.F2MAX-f2)/(VP.F2MAX-VP.F2MIN)*(Wp-VP.pad*2);}
function vY(f1,H){return VP.pad+(f1-VP.F1MIN)/(VP.F1MAX-VP.F1MIN)*(H-VP.pad*2);}
function drawVowel(t){const g=vowelC.getContext("2d"),Wp=vowelC.width,H=vowelC.height;g.clearRect(0,0,Wp,H);
  g.strokeStyle="#21252e";g.strokeRect(VP.pad,VP.pad,Wp-VP.pad*2,H-VP.pad*2);
  g.fillStyle="#6b7280";g.font="10px system-ui";
  g.fillText("← F2 (front)",VP.pad,14);g.fillText("(back) →",Wp-58,14);
  g.save();g.translate(11,H/2);g.rotate(-Math.PI/2);g.fillText("F1 (close → open)",-40,0);g.restore();
  g.font="11px system-ui";
  for(const v of VOWELS){const x=vX(v.F2,Wp),y=vY(v.F1,H);g.strokeStyle="#8b93a3";g.beginPath();g.arc(x,y,3,0,7);g.stroke();
    g.fillStyle="#9aa3b2";g.fillText(v.s+" "+v.w,x+5,y+3);}
  const i=idxAt(t);
  for(let k=Math.max(0,i-22);k<=i;k++){const f1=clip.F1[k],f2=clip.F2[k];if(f1==null||f2==null)continue;
    const a=(k-(i-22))/22;g.fillStyle="rgba(124,156,255,"+(0.12+0.45*a)+")";g.beginPath();g.arc(vX(f2,Wp),vY(f1,H),2.2,0,7);g.fill();}
  const f1=clip.F1[i],f2=clip.F2[i];
  if(f1!=null&&f2!=null){const nv=nearestVowel(f1,f2);g.fillStyle="#fff";g.strokeStyle=css("--accent");g.lineWidth=2.5;
    g.beginPath();g.arc(vX(f2,Wp),vY(f1,H),6,0,7);g.fill();g.stroke();g.lineWidth=1;
    g.fillStyle=css("--accent");g.font="12px system-ui";g.fillText("≈ "+nv.s+" ("+nv.w+")",vX(f2,Wp)+9,vY(f1,H)-7);}
  else{g.fillStyle="#6b7280";g.font="11px system-ui";g.fillText("(unvoiced — no vowel here)",VP.pad+6,VP.pad+15);}}

function setCursorEls(){
  playhead.style.left=(8+playT/clip.dur*W)+"px";
  if(hoverT!=null&&audio.paused){hoverEl.style.display="block";hoverEl.style.left=(8+hoverT/clip.dur*W)+"px";}
  else hoverEl.style.display="none";
  if(sel){selEl.style.display="block";selEl.style.left=(8+sel.a/clip.dur*W)+"px";selEl.style.width=((sel.b-sel.a)/clip.dur*W)+"px";}
  else selEl.style.display="none";}
function updateReadout(t){const i=idxAt(t),f0=clip.f0[i],F1=clip.F1[i],F2=clip.F2[i],F3=clip.F3[i],v=x=>x==null?"—":Math.round(x);
  readout.innerHTML=`t <b>${t.toFixed(2)}s</b> | <span class="chip" style="background:var(--pitch)">F0</span> <b>${f0==null?"— unvoiced":v(f0)+" Hz"}</b> `+
    `<span class="chip" style="background:var(--f1)">F1</span> <b>${v(F1)}</b> `+
    `<span class="chip" style="background:var(--f2)">F2</span> <b>${v(F2)}</b> `+
    `<span class="chip" style="background:var(--f3)">F3</span> <b>${v(F3)}</b>`;}
function refresh(){const t=activeT();setCursorEls();drawSlice(t);drawVowel(t);updateReadout(t);}

function tAtX(clientX){const r=stage.getBoundingClientRect();return Math.max(0,Math.min(clip.dur,(clientX-r.left-8)/W*clip.dur));}
stage.addEventListener("mousedown",e=>{down=true;moved=false;downX=e.clientX;});
stage.addEventListener("mousemove",e=>{
  if(down){if(Math.abs(e.clientX-downX)>4){moved=true;const a=tAtX(downX),b=tAtX(e.clientX);sel={a:Math.min(a,b),b:Math.max(a,b)};setCursorEls();}}
  else{hoverT=tAtX(e.clientX);if(audio.paused)refresh();else{hoverEl.style.display="block";hoverEl.style.left=(8+hoverT/clip.dur*W)+"px";}}});
stage.addEventListener("mouseup",e=>{if(!down)return;down=false;
  if(!moved){playT=tAtX(e.clientX);audio.currentTime=playT;hoverT=null;refresh();}
  else if(sel&&sel.b-sel.a<0.04){sel=null;setCursorEls();}});
stage.addEventListener("mouseleave",()=>{hoverT=null;hoverEl.style.display="none";if(audio.paused)refresh();});

function setPlay(){$("#play").textContent=audio.paused?"▶ Play":"⏸ Pause";}
$("#play").onclick=()=>{if(audio.paused){if(sel&&(audio.currentTime<sel.a||audio.currentTime>=sel.b))audio.currentTime=sel.a;audio.play();}else audio.pause();};
$("#loop").onclick=()=>{looping=!looping;$("#loop").classList.toggle("on",looping);};
$("#clearsel").onclick=()=>{sel=null;looping=false;$("#loop").classList.remove("on");setCursorEls();};
$("#speed").querySelectorAll("button").forEach(b=>b.onclick=()=>setSpeed(+b.dataset.s));
audio.addEventListener("play",setPlay);audio.addEventListener("pause",setPlay);audio.addEventListener("ended",setPlay);
document.addEventListener("keydown",e=>{if(e.code==="Space"&&e.target.tagName!=="BUTTON"){e.preventDefault();$("#play").onclick();}});
specImg.addEventListener("load",()=>{if(clip)drawFormants();});
window.addEventListener("resize",()=>{if(clip){layout();refresh();}});
function frame(){if(clip&&!audio.paused){if(looping&&sel&&audio.currentTime>=sel.b-0.005)audio.currentTime=sel.a;
  playT=audio.currentTime;refresh();}requestAnimationFrame(frame);}

buildClips();select(0);requestAnimationFrame(frame);
</script></body></html>
"""

if __name__ == "__main__":
    main()
