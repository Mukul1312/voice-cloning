"""
build_blind_clone_ab.py — a BLIND, RECORDED two-part ear-test: original-data clone vs denoised-data clone.

ECAPA says denoising slightly hurt (best unseen 0.743 -> 0.704/0.727). Ear and ECAPA diverged once before
(the denoise gate), so test it properly and blind:
  PART 1 — ABX discrimination: hear A, B, then X; is X==A or X==B? (can you even tell them apart?)
  PART 2 — Preference: which is "more him" + "more natural", with confidence + what-differed tags.
Clips are loudness-matched (so volume can't bias), relabeled A/B (randomized, opaque filenames), and every
measurement I have is hidden in a base64 key — revealed only when the listener hits Finish, alongside the
spectrograms and an ear<->ECAPA agreement read. Results export to blind_results.json.

Reuses scripts/eval_metrics.py (ECAPA+MCD+F0), the spec_png approach from build_denoise_compare.py, and the
blind/base64/export conventions from build_blind_gallery.py.
Output -> voice_lab/blind_clone_ab/   Run -> .venv/Scripts/python.exe scripts/build_blind_clone_ab.py
"""
import base64
import importlib.util
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL_ORIG = ROOT / "data/out/lecture1/eval"
EVAL_DN = ROOT / "data/out/lecture1/eval_dn"
FINAL = ROOT / "data/out/lecture1/final"
CLIPS = FINAL / "clips"
CLIPS_DN = FINAL / "clips_dn"
OUT = ROOT / "voice_lab" / "blind_clone_ab"
AUD = OUT / "audio"
IMG = OUT / "img"

# scoring engine (loads ECAPA + build_voice_anatomy once; main() is __name__-guarded)
_spec = importlib.util.spec_from_file_location("em", ROOT / "scripts" / "eval_metrics.py")
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CKPT = "150"
PREF_SEEDS = [42, 123, 7]      # 3 sentences x 3 seeds = 9 preference pairs
ABX_SEEDS = [42, 123]          # 3 sentences x 2 seeds = 6 ABX trials
REAL_NM = CLIPS / "ggs_l1_0112.wav"
TARGET_RMS = 0.06
SENTS = {
    "nm":  {"seen": True,  "text": "Namabhasa means offences are not completely gone. If offences will completely go, then pure name will rise"},
    "nv1": {"seen": False, "text": "Without the mercy of guru and Krishna, no one can cross this ocean of birth and death."},
    "nv2": {"seen": False, "text": "The pure devotee always remembers Krishna and never forgets Him for a single moment."},
}
RNG = random.Random(20260630)


def rms_match(src, dst):
    """Normalize a clip to a common RMS (with peak guard) so loudness can't bias the listener."""
    y, sr = sf.read(str(src), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    r = float(np.sqrt(np.mean(y * y))) + 1e-9
    y = y * (TARGET_RMS / r)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y = y * (0.99 / pk)
    sf.write(str(dst), y.astype("float32"), sr)


def spec_png(wav_path, out_png):
    """0-8 kHz dB spectrogram (same approach as build_denoise_compare.spec_png), for the reveal."""
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


def score_pair(orig, dn, sent):
    eo, ed = em.embed(orig), em.embed(dn)
    so = {"ecapa": round(em.cosine(eo, ref_noisy), 3), "f0": em.f0_median(orig)}
    sd = {"ecapa_noisy": round(em.cosine(ed, ref_noisy), 3),
          "ecapa_dn": round(em.cosine(ed, ref_dn), 3), "f0": em.f0_median(dn)}
    if sent == "nm":
        so["mcd"] = round(em.mcd(orig, REAL_NM), 2)
        sd["mcd"] = round(em.mcd(dn, REAL_NM), 2)
        fo, fd = em.f0_rmse_semitones(orig, REAL_NM), em.f0_rmse_semitones(dn, REAL_NM)
        so["f0rmse"] = round(fo, 2) if fo else None
        sd["f0rmse"] = round(fd, 2) if fd else None
    so["f0"] = round(so["f0"], 0) if so["f0"] else None
    sd["f0"] = round(sd["f0"], 0) if sd["f0"] else None
    diff = so["ecapa"] - sd["ecapa_noisy"]
    fav = "orig" if diff > 0.01 else ("dn" if diff < -0.01 else "tie")
    return so, sd, fav


def cell(sent, seed):
    fn = f"c{CKPT}__{sent}__s{seed}.wav"
    return EVAL_ORIG / fn, EVAL_DN / fn


# ---------- build ----------
if OUT.exists():
    shutil.rmtree(OUT)
AUD.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

# centroids (same selection as score_denoise_retrain.py)
rows = [json.loads(l) for l in (FINAL / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
eng = set((ROOT / "data/english_words.txt").read_text(encoding="utf-8").split())
tok = re.compile(r"[a-z']+")
def ef(t):
    w = tok.findall(t.lower()); return (sum(1 for x in w if x in eng) / len(w)) if w else 0
EXCL = {"ggs_l1_0112", "ggs_l1_0114"}
sel = [Path(r["audio"]).name for r in rows if 5 <= r["duration"] <= 12 and ef(r["text"]) >= 0.85 and Path(r["audio"]).stem not in EXCL]
ref_names = sel[:20]
ref_noisy = em.centroid([CLIPS / n for n in ref_names])
ref_dn = em.centroid([CLIPS_DN / n for n in ref_names])
ceil_noisy = round(float(np.mean([em.cosine(em.embed(CLIPS / n), ref_noisy) for n in sel[20:30]])), 3)

# real anchors (labelled, loudness-matched, not blind)
anchors = []
for name, lab in [("ggs_l1_0112.wav", "real — Namabhasa (same words as the SEEN sentence)"),
                  (sel[20], "real — English sample 1"), (sel[21], "real — English sample 2")]:
    if (CLIPS / name).exists():
        rms_match(CLIPS / name, AUD / f"real_{name}")
        anchors.append({"wav": f"audio/real_{name}", "label": lab})

# ----- PART 1: ABX trials -----
abx_vis, abx_key = [], {}
xi = 0
for sent in ("nm", "nv1", "nv2"):
    for seed in ABX_SEEDS:
        o, d = cell(sent, seed)
        if not (o.exists() and d.exists()):
            continue
        xi += 1
        xid = f"x{xi}"
        a_is_orig = RNG.random() < 0.5
        aSrc, bSrc = (o, d) if a_is_orig else (d, o)
        rms_match(aSrc, AUD / f"{xid}_A.wav")
        rms_match(bSrc, AUD / f"{xid}_B.wav")
        x_is_a = RNG.random() < 0.5
        shutil.copyfile(AUD / f"{xid}_{'A' if x_is_a else 'B'}.wav", AUD / f"{xid}_X.wav")
        abx_vis.append({"id": xid, "text": SENTS[sent]["text"], "seen": SENTS[sent]["seen"],
                        "A": f"audio/{xid}_A.wav", "B": f"audio/{xid}_B.wav", "X": f"audio/{xid}_X.wav"})
        abx_key[xid] = {"x_is_a": x_is_a, "A": "orig" if a_is_orig else "dn",
                        "B": "dn" if a_is_orig else "orig", "sent": sent, "seed": seed}

# ----- PART 2: preference pairs (+ 1 attention check) -----
pref_vis, pref_key = [], {}
pi = 0
for sent in ("nm", "nv1", "nv2"):
    for seed in PREF_SEEDS:
        o, d = cell(sent, seed)
        if not (o.exists() and d.exists()):
            continue
        pi += 1
        pid = f"p{pi}"
        so, sd, fav = score_pair(o, d, sent)
        a_is_orig = RNG.random() < 0.5
        aSrc, bSrc = (o, d) if a_is_orig else (d, o)
        rms_match(aSrc, AUD / f"{pid}_A.wav")
        rms_match(bSrc, AUD / f"{pid}_B.wav")
        spec_png(AUD / f"{pid}_A.wav", IMG / f"{pid}_A.png")
        spec_png(AUD / f"{pid}_B.wav", IMG / f"{pid}_B.png")
        pref_vis.append({"id": pid, "text": SENTS[sent]["text"], "seen": SENTS[sent]["seen"],
                         "A": f"audio/{pid}_A.wav", "B": f"audio/{pid}_B.wav",
                         "specA": f"img/{pid}_A.png", "specB": f"img/{pid}_B.png", "attention": False})
        pref_key[pid] = {"A": "orig" if a_is_orig else "dn", "B": "dn" if a_is_orig else "orig",
                         "sent": sent, "seed": seed, "attention": False,
                         "scores": {"orig": so, "dn": sd}, "fav": fav}

# attention-check: A and B are the SAME clip -> correct answer is "Same"
ao, _ = cell("nv2", PREF_SEEDS[1])
if ao.exists():
    pi += 1
    pid = f"p{pi}"
    rms_match(ao, AUD / f"{pid}_A.wav")
    shutil.copyfile(AUD / f"{pid}_A.wav", AUD / f"{pid}_B.wav")
    spec_png(AUD / f"{pid}_A.wav", IMG / f"{pid}_A.png")
    shutil.copyfile(IMG / f"{pid}_A.png", IMG / f"{pid}_B.png")
    pref_vis.append({"id": pid, "text": SENTS["nv2"]["text"], "seen": False,
                     "A": f"audio/{pid}_A.wav", "B": f"audio/{pid}_B.wav",
                     "specA": f"img/{pid}_A.png", "specB": f"img/{pid}_B.png", "attention": True})
    pref_key[pid] = {"A": "orig", "B": "orig", "attention": True}

RNG.shuffle(pref_vis)  # randomize order (truth stays keyed by id)

VISIBLE = {"ceil": ceil_noisy, "anchors": anchors, "abx": abx_vis, "pairs": pref_vis,
           "tags": ["muffled", "robotic", "breathy", "glitch/warble", "pitch off", "pacing off", "clearer", "none"],
           "verdict": {"orig_best": 0.743, "dn_best_noisy": 0.704, "dn_best_dn": 0.727}}
KEY_B64 = base64.b64encode(json.dumps({"abx": abx_key, "pairs": pref_key}).encode()).decode()


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind ear-test — clone A/B</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;--real:#7c9cff;--a:#46c08a;--b:#ff9d4a;--ok:#46c08a;--no:#ff5d5d;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:940px;margin:0 auto;padding:18px 16px 110px}
  h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 8px;border-bottom:1px solid var(--line);padding-bottom:5px}
  .sub{color:var(--mut);margin:0 0 12px}
  details.notes{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 14px;margin:0 0 14px}
  details.notes summary{cursor:pointer;font-weight:700}.notes ol{margin:8px 0 0;padding-left:20px}.notes li{margin:4px 0}.notes .q{color:var(--real);font-weight:700}
  .ref{background:var(--card);border:1px solid var(--real);border-radius:12px;padding:10px 14px;margin:0 0 12px}
  .ref .t{color:var(--real);font-weight:700;margin:0 0 6px}.rrow{display:flex;gap:10px;align-items:center;margin:3px 0;font-size:12.5px;color:var(--mut)}.rrow audio{height:32px;flex:1;max-width:360px}
  .prog{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;font-size:13px;color:var(--mut);border-bottom:1px solid var(--line);margin-bottom:10px}
  .item{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:0 0 12px}
  .item.active{border-color:var(--real)}
  .ph{font-size:13px;color:var(--mut);margin:0 0 2px}.ptext{margin:0 0 9px;font-size:13.5px}
  .seen{font-size:11px;padding:1px 7px;border-radius:20px;background:#3a2e1a;color:#ffb84a;margin-left:6px}.uns{font-size:11px;padding:1px 7px;border-radius:20px;background:#16302a;color:#46c08a;margin-left:6px}
  .clips{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 8px}.cc{flex:1 1 260px}
  .cl{font-weight:700;margin:0 0 3px}.cl.a{color:var(--a)}.cl.b{color:var(--b)}.cl.x{color:var(--real)}
  audio{width:100%;height:34px}.tools{font-size:11px;color:var(--mut);margin-top:2px}.tools a{color:var(--real);cursor:pointer;margin-right:8px}
  .qrow{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}.qlab{font-size:12.5px;color:var(--mut);min-width:96px}
  .opt{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--ink);border-radius:8px;padding:5px 11px;font:inherit;font-weight:700}
  .opt.A.on{background:var(--a);color:#0f1115;border-color:var(--a)}.opt.B.on{background:var(--b);color:#0f1115;border-color:var(--b)}.opt.on:not(.A):not(.B){background:var(--mut);color:#0f1115}
  .opt:disabled{opacity:.4;cursor:not-allowed}
  .tags{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0}.tag{cursor:pointer;border:1px solid var(--line);background:#11141b;color:var(--mut);border-radius:20px;padding:3px 10px;font-size:12px}.tag.on{background:#2a2e38;color:var(--ink)}
  textarea{width:100%;margin-top:6px;background:#11141b;border:1px solid var(--line);color:var(--ink);border-radius:8px;padding:6px 8px;font:inherit;font-size:12.5px;resize:vertical}
  .gate{font-size:11.5px;color:var(--no)}.gate.ok{color:var(--ok)}
  .reveal{margin-top:8px;border-top:1px dashed var(--line);padding-top:8px;font-size:12.5px;color:var(--mut);display:none}.reveal.show{display:block}.reveal b{color:var(--ink)}
  .reveal img{width:48%;border-radius:6px;background:#000;margin-top:4px}.agree{color:var(--ok);font-weight:700}.diverge{color:var(--no);font-weight:700}
  .bar{position:fixed;left:0;right:0;bottom:0;background:#11141b;border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center}
  .bar button{cursor:pointer;border:1px solid var(--real);background:var(--real);color:#0f1115;border-radius:8px;padding:8px 16px;font:inherit;font-weight:800}.bar button.ghost{background:#11141b;color:var(--ink)}
  #summary{background:var(--card);border:1px solid var(--real);border-radius:12px;padding:12px 14px;margin:0 0 12px;display:none}#summary.show{display:block}#out{display:none;width:100%;height:130px;margin-top:8px}
</style></head><body><div class="wrap">
  <h1>🎧 Blind ear-test — which clone is more <em>him</em>?</h1>
  <p class="sub">Two clones, <b style="color:var(--a)">A</b> vs <b style="color:var(--b)">B</b>, same words & seed, <b>loudness-matched</b>. One trained on original audio, one on denoised — <b>you won't know which.</b> My measurements stay hidden until you finish.</p>
  <details class="notes" open><summary>How to do this (read once)</summary><ol>
    <li>Play <b style="color:var(--real)">his real voice</b> below once — get the feel of "that's him."</li>
    <li><b>Part 1 (ABX):</b> hear A, B, then X — decide if X is the same clip as A or B. (Tests if you can tell them apart at all.)</li>
    <li><b>Part 2 (Preference):</b> for each pair, <span class="q">"which is more him, and which is more natural — or are they the same?"</span> Add a confidence and tick any difference you noticed.</li>
    <li><b>"Same / can't tell" is a totally valid answer.</b> You must play the clips before you can answer.</li>
    <li>When done, hit <b>Finish &amp; reveal</b>. Keys: click a card then press <b>1/2/3</b> for A/Same/B.</li>
  </ol></details>
  <div class="ref" id="ref"></div>
  <div class="prog" id="prog"></div>
  <div id="summary"></div>
  <h2>Part 1 — can you tell them apart? (ABX)</h2><div id="abxlist"></div>
  <h2>Part 2 — which is more him? (preference)</h2><div id="preflist"></div>
</div>
<div class="bar"><button id="finish">Finish &amp; reveal ▸</button><button class="ghost" id="copy" style="display:none">Copy results to send Claude</button></div>
<textarea id="out" readonly></textarea>
<script>
const DATA = /*__DATA__*/;
const KEY_B64 = "__KEY_B64__";
const $=s=>document.querySelector(s), ce=(t,c)=>{const e=document.createElement(t);if(c)e.className=c;return e;};
const LSKEY="blind_clone_ab_v2";
let st=JSON.parse(localStorage.getItem(LSKEY)||"{}");
st.abx=st.abx||{}; st.pref=st.pref||{};
const played={}; // id -> Set of clips played (force-listen, in-memory)
let active=null;
const save=()=>localStorage.setItem(LSKEY,JSON.stringify(st));

$("#ref").innerHTML=`<div class="t">🔵 his real voice (anchor)</div>`+DATA.anchors.map(a=>
  `<div class="rrow"><span>${a.label}</span><audio controls preload="none" src="${a.wav}"></audio></div>`).join("");

function mkAudio(id,clip,src){const a=ce("audio");a.controls=true;a.preload="none";a.src=src;
  a.addEventListener("play",()=>{(played[id]=played[id]||new Set()).add(clip);refreshGate(id);});return a;}
function tools(a){const w=ce("div","tools");
  const r=ce("a");r.textContent="↻ replay";r.onclick=()=>{a.currentTime=0;a.play();};
  const s=ce("a");s.textContent="0.75×";s.onclick=()=>{a.playbackRate=a.playbackRate===1?0.75:1;s.textContent=a.playbackRate===1?"0.75×":"1×";};
  w.append(r,s);return w;}

// ---------- Part 1: ABX ----------
const abxBox=$("#abxlist");
DATA.abx.forEach((t,i)=>{const d=ce("div","item");d.id="item_"+t.id;d.onclick=()=>setActive(t.id);
  d.innerHTML=`<div class="ph">ABX ${i+1} of ${DATA.abx.length}${t.seen?'<span class="seen">SEEN</span>':'<span class="uns">UNSEEN</span>'}</div><div class="ptext">“${t.text}”</div>`;
  const clips=ce("div","clips");
  [["A",t.A],["B",t.B],["X",t.X]].forEach(([lab,src])=>{const c=ce("div","cc");const l=ce("div","cl "+lab.toLowerCase());l.textContent="▶ "+lab;
    const au=mkAudio(t.id,lab,src);c.append(l,au,tools(au));clips.appendChild(c);});
  d.appendChild(clips);
  const q=ce("div","qrow");q.innerHTML='<span class="qlab">X is the same as…</span>';
  [["A","X = A"],["B","X = B"],["noidea","no idea"]].forEach(([v,txt])=>{const b=ce("button","opt "+(v==="A"?"A":v==="B"?"B":"S"));
    b.textContent=txt;b.dataset.v=v;b.disabled=true;b.onclick=()=>{st.abx[t.id]={pick:v};save();markPick(q,b);prog();};
    if((st.abx[t.id]||{}).pick===v)b.classList.add("on");q.appendChild(b);});
  d.appendChild(q);const g=ce("div","gate");g.id="gate_"+t.id;g.textContent="▶ play A, B and X to unlock";d.appendChild(g);
  abxBox.appendChild(d);});

// ---------- Part 2: preference ----------
const prefBox=$("#preflist");
DATA.pairs.forEach((p,i)=>{const d=ce("div","item");d.id="item_"+p.id;d.onclick=()=>setActive(p.id);
  d.innerHTML=`<div class="ph">Pair ${i+1} of ${DATA.pairs.length}${p.seen?'<span class="seen">SEEN</span>':'<span class="uns">UNSEEN</span>'}</div><div class="ptext">“${p.text}”</div>`;
  const clips=ce("div","clips");
  [["A",p.A],["B",p.B]].forEach(([lab,src])=>{const c=ce("div","cc");const l=ce("div","cl "+lab.toLowerCase());l.textContent="▶ "+lab;
    const au=mkAudio(p.id,lab,src);c.append(l,au,tools(au));clips.appendChild(c);});
  d.appendChild(clips);
  const a=st.pref[p.id]||{};
  function qrow(field,label){const q=ce("div","qrow");q.innerHTML=`<span class="qlab">${label}</span>`;
    [["A","A"],["same","Same"],["B","B"]].forEach(([v,txt])=>{const b=ce("button","opt "+(v==="A"?"A":v==="B"?"B":"S"));
      b.textContent=txt;b.dataset.v=v;b.dataset.field=field;b.disabled=true;
      b.onclick=()=>{st.pref[p.id]={...(st.pref[p.id]||{}),[field]:v};save();markPick(q,b);prog();};
      if((st.pref[p.id]||{})[field]===v)b.classList.add("on");q.appendChild(b);});return q;}
  d.appendChild(qrow("him","more HIM"));
  d.appendChild(qrow("nat","more NATURAL"));
  // confidence
  const cf=ce("div","qrow");cf.innerHTML='<span class="qlab">confidence</span>';
  [["sure","sure"],["unsure","unsure"],["guess","guess"]].forEach(([v,txt])=>{const b=ce("button","opt S");b.textContent=txt;b.dataset.v=v;b.disabled=true;
    b.onclick=()=>{st.pref[p.id]={...(st.pref[p.id]||{}),conf:v};save();markPick(cf,b);};if((a).conf===v)b.classList.add("on");cf.appendChild(b);});
  d.appendChild(cf);
  // tags
  const tg=ce("div","tags");DATA.tags.forEach(t=>{const b=ce("span","tag"+(((a.tags||[]).includes(t))?" on":""));b.textContent=t;
    b.onclick=()=>{const cur=new Set((st.pref[p.id]||{}).tags||[]);cur.has(t)?cur.delete(t):cur.add(t);
      st.pref[p.id]={...(st.pref[p.id]||{}),tags:[...cur]};save();b.classList.toggle("on");};tg.appendChild(b);});
  d.appendChild(tg);
  const ta=ce("textarea");ta.placeholder="(optional) note";ta.value=a.note||"";
  ta.oninput=()=>{st.pref[p.id]={...(st.pref[p.id]||{}),note:ta.value};save();};d.appendChild(ta);
  const g=ce("div","gate");g.id="gate_"+p.id;g.textContent="▶ play A and B to unlock";d.appendChild(g);
  const rv=ce("div","reveal");rv.id="rev_"+p.id;d.appendChild(rv);
  prefBox.appendChild(d);});

function markPick(row,btn){row.querySelectorAll(".opt").forEach(x=>{if(x.dataset.field===btn.dataset.field||(!x.dataset.field&&!btn.dataset.field&&row===x.parentElement))x.classList.remove("on");});
  // simpler: clear siblings in same row sharing field
  row.querySelectorAll(".opt").forEach(x=>{if((x.dataset.field||"")===(btn.dataset.field||""))x.classList.remove("on");});btn.classList.add("on");}
function need(id){return DATA.abx.find(t=>t.id===id)?["A","B","X"]:["A","B"];}
function refreshGate(id){const ok=need(id).every(c=>(played[id]||new Set()).has(c));
  const g=$("#gate_"+id);if(g){g.textContent=ok?"✓ unlocked":("▶ play "+need(id).join(", ")+" to unlock");g.classList.toggle("ok",ok);}
  document.querySelectorAll(`#item_${id} .opt`).forEach(b=>b.disabled=!ok);}
function setActive(id){active=id;document.querySelectorAll(".item").forEach(e=>e.classList.toggle("active",e.id==="item_"+id));}
function answered(){let n=0;DATA.abx.forEach(t=>{if((st.abx[t.id]||{}).pick)n++;});DATA.pairs.forEach(p=>{if((st.pref[p.id]||{}).him)n++;});return n;}
function prog(){const tot=DATA.abx.length+DATA.pairs.length;$("#prog").textContent=`Answered ${answered()} / ${tot}`+(answered()>=tot?"  ✓ ready to finish":"  — ABX pick + 'more him' on each pair");}

// keyboard: 1/2/3 -> A/Same/B on the active card's primary question
document.addEventListener("keydown",e=>{if(!active||["TEXTAREA","INPUT"].includes(e.target.tagName))return;
  const map={"1":"A","2":"same","3":"B"};if(!(e.key in map))return;const v=map[e.key];
  if(DATA.abx.find(t=>t.id===active)){const vv=v==="same"?"noidea":v;const b=document.querySelector(`#item_${active} .opt[data-v="${vv}"]`);if(b&&!b.disabled)b.click();}
  else{const b=document.querySelector(`#item_${active} .opt[data-field="him"][data-v="${v}"]`);if(b&&!b.disabled)b.click();}});

DATA.abx.forEach(t=>refreshGate(t.id));DATA.pairs.forEach(p=>refreshGate(p.id));prog();

function reveal(){const K=JSON.parse(atob(KEY_B64));
  // ABX accuracy
  let abxC=0,abxN=0;DATA.abx.forEach(t=>{const a=(st.abx[t.id]||{}).pick;if(!a||a==="noidea")return;abxN++;
    const correct=(a==="A")===K.abx[t.id].x_is_a;if(correct)abxC++;});
  // preference
  let earOrig=0,earDn=0,same=0,metOrig=0,agree=0,scored=0,attnOK=null;
  DATA.pairs.forEach(p=>{const kk=K.pairs[p.id];const a=st.pref[p.id]||{};const r=$("#rev_"+p.id);r.classList.add("show");
    if(kk.attention){attnOK=(a.him==="same");
      r.innerHTML=`<b>Attention-check</b> (A and B were the SAME clip). You said: <b>${a.him||'—'}</b> → ${attnOK?'<span class="agree">good, you correctly heard them as identical</span>':'<span class="diverge">you picked a winner on identical clips — judgments are noisy here</span>'}`;return;}
    const earPick=a.him==="A"?kk.A:a.him==="B"?kk.B:"same";
    if(earPick==="orig")earOrig++;else if(earPick==="dn")earDn++;else same++;
    if(kk.fav==="orig")metOrig++;let ag="";
    if(a.him&&a.him!=="same"){scored++;if(kk.fav==="tie")ag='<span class="agree">~tie</span>';
      else if(earPick===kk.fav){agree++;ag='<span class="agree">ear agrees with ECAPA</span>';}else ag='<span class="diverge">ear diverges from ECAPA</span>';}
    const so=kk.scores.orig,sd=kk.scores.dn;const mcd=so.mcd!=null?` · MCD orig ${so.mcd}/dn ${sd.mcd} · F0-RMSE orig ${so.f0rmse}/dn ${sd.f0rmse}`:'';
    const sA=p.A.includes("_A")?p.specA:p.specB; // A spectrogram
    r.innerHTML=`<b>A = ${kk.A}</b>, <b>B = ${kk.B}</b>. ECAPA(vs noisy): orig <b>${so.ecapa}</b> · dn <b>${sd.ecapa_noisy}</b> (dn-centroid ${sd.ecapa_dn}) · F0 orig ${so.f0}/dn ${sd.f0}Hz${mcd}. You: him <b>${a.him||'—'}</b> (${a.conf||'—'}) → ${a.him==='same'?'same':earPick}. ${ag}<br><img src="${p.specA}" title="A"><img src="${p.specB}" title="B">`;});
  const s=$("#summary");s.classList.add("show");
  s.innerHTML=`<b>📊 Your blind verdict</b><br>`+
    `<b>ABX:</b> ${abxC}/${abxN} correct (${abxN?Math.round(100*abxC/abxN):0}%). ${abxN&&abxC/abxN<0.65?'<span class="agree">≈ chance — the two are hard to tell apart at all.</span>':abxN?'you could often tell them apart.':''}<br>`+
    `<b>Preference:</b> ear chose <b style="color:var(--a)">original ${earOrig}</b> · <b style="color:var(--b)">denoised ${earDn}</b> · same ${same}. ECAPA favored original in ${metOrig}. Ear↔ECAPA agreed <b>${agree}/${scored}</b>.<br>`+
    (attnOK===null?'':`<b>Attention-check:</b> ${attnOK?'<span class="agree">passed</span>':'<span class="diverge">missed</span>'}.<br>`)+
    `<span style="color:var(--mut)">Now hit "Copy results" and paste them to Claude for the full read.</span>`;
  window.scrollTo(0,0);$("#copy").style.display="";$("#out").style.display="block";
  $("#out").value=JSON.stringify({lskey:LSKEY,ceil:DATA.ceil,
    abx:DATA.abx.map(t=>({id:t.id,sent:K.abx[t.id].sent,seed:K.abx[t.id].seed,pick:(st.abx[t.id]||{}).pick||null,truth_x_is_a:K.abx[t.id].x_is_a,correct:((st.abx[t.id]||{}).pick==="A")===K.abx[t.id].x_is_a})),
    pairs:DATA.pairs.map(p=>{const kk=K.pairs[p.id];const a=st.pref[p.id]||{};return{id:p.id,attention:!!kk.attention,sent:kk.sent,seed:kk.seed,truth:{A:kk.A,B:kk.B},him:a.him||null,natural:a.nat||null,conf:a.conf||null,tags:a.tags||[],note:a.note||"",ear:kk.attention?null:(a.him==="A"?kk.A:a.him==="B"?kk.B:"same"),metric_favors:kk.fav||null,scores:kk.scores||null};})},null,2);}
$("#finish").onclick=reveal;
$("#copy").onclick=()=>{$("#out").select();document.execCommand("copy");$("#copy").textContent="✓ copied — paste to Claude";};
</script></body></html>
"""

html = (TEMPLATE.replace("/*__DATA__*/", json.dumps(VISIBLE, ensure_ascii=False))
                .replace("__KEY_B64__", KEY_B64))
(OUT / "index.html").write_text(html, encoding="utf-8")
print(f"wrote {OUT/'index.html'}  ({len(pref_vis)} preference pairs, {len(abx_vis)} ABX trials)")
