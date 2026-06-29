#!/usr/bin/env python
# =====================================================================================
# build_qa_gallery.py — turn a processed lecture dir into a self-contained listening-QA gallery.
#
# Joins qa_asr_report.tsv (authoritative 134-row QA: lead/trail/status/reasons/metrics) with
# train.jsonl (transcripts) by clip index, renders a mel-spectrogram + waveform thumbnail per clip
# (the L18 recipe: melspectrogram -> power_to_db), and emits ONE static qa_gallery.html.
#
# The html needs NO server: <img>/<audio> use relative paths, decisions persist in localStorage,
# export is a Blob download. Just double-click qa_gallery.html.
#
# Run with the venv that has librosa+matplotlib:
#   .venv/Scripts/python scripts/build_qa_gallery.py --dir data/out/lecture1
#   .venv/Scripts/python scripts/build_qa_gallery.py --dir data/out/lecture1 --force   # re-render thumbs
# =====================================================================================
import argparse, csv, json, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

IDX_RE = re.compile(r"_(\d+)\.wav$")


def clip_index(audio_path: str) -> int:
    m = IDX_RE.search(audio_path)
    return int(m.group(1)) if m else -1


def render_thumb(wav: Path, png: Path):
    """FULL-BLEED waveform strip over a mel-spectrogram, both spanning EXACTLY 0..duration edge-to-edge
    with zero margins. Two payoffs: (a) the waveform and spectrogram are pixel-aligned in x (a known
    tight_layout+sharex misalignment is avoided by removing all in-image axis decorations), and (b) the
    html overlay can map pixel<->time as a single linear ratio, giving a millisecond playhead + hover
    readout. Mel recipe is L18: melspectrogram -> power_to_db(ref=max); hop 256 matches our silence gate."""
    y, sr = librosa.load(str(wav), sr=16000)
    if len(y) == 0:
        y = np.zeros(sr // 10, dtype=np.float32)
    # higher time/freq resolution (hop 128, 160 mels, n_fft 2048) so the cells are small; the
    # imshow(interpolation="bilinear") below then shades them smoothly instead of as flat blocks.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=128, n_mels=160)
    logmel = librosa.power_to_db(mel, ref=np.max)
    dur = len(y) / sr

    fig = plt.figure(figsize=(9.0, 2.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.8], hspace=0.0,
                          left=0.0, right=1.0, top=1.0, bottom=0.0)  # axes fill the whole canvas
    axw = fig.add_subplot(gs[0])
    axm = fig.add_subplot(gs[1], sharex=axw)

    t = np.arange(len(y)) / sr
    axw.axhline(0, lw=0.3, color="#22303c")
    axw.plot(t, y, lw=0.5, color="#5cc8ff")
    axw.set_ylim(-1.05, 1.05)
    axw.set_facecolor("#0d1117")
    axw.axis("off")

    axm.imshow(logmel, origin="lower", aspect="auto", extent=[0, dur, 0, logmel.shape[0]],
               cmap="magma", interpolation="bilinear")   # smooth shading + exact 0..dur extent
    axm.set_ylim(0, logmel.shape[0])
    axm.set_facecolor("#0d1117")
    axm.axis("off")

    axw.set_xlim(0, dur)   # sharex -> both axes locked to 0..dur across the full width
    fig.patch.set_facecolor("#0d1117")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png), dpi=200, facecolor="#0d1117", pad_inches=0)  # 2x res: stays crisp on hi-DPI / when upscaled; no bbox='tight' -> full-bleed
    plt.close(fig)
    return round(float(dur), 2)


def load_records(d: Path):
    # transcripts by clip index
    texts = {}
    tj = d / "train.jsonl"
    if tj.exists():
        for line in tj.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            texts[clip_index(o["audio"])] = o.get("text", "")
    # QA report is the authoritative clip list
    rows = []
    with open(d / "qa_asr_report.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["clip"])
            rows.append({
                "idx": i,
                "file": f"clips/i-and-mine-and-namabhasa-stage_{i:04d}.wav",
                "thumb": f"_thumbs/{i:04d}.png",
                "text": texts.get(i, ""),
                "dur": float(r["dur"]),
                "align_min": float(r["align_min"]),
                "align_mean": float(r["align_mean"]),
                "cer_core": float(r["cer_core"]),
                "cer_raw": float(r["cer_raw"]),
                "wer": float(r["wer"]),
                "lead": float(r["lead"]),
                "trail": float(r["trail"]),
                "status": r["status"].strip(),
                "reasons": r["reasons"].strip(),
            })
    return rows


def fix_filenames(d: Path, rows):
    """The slug prefix is hard-coded above; verify the wavs actually exist and repair the prefix
    from whatever is on disk if the lecture slug differs."""
    clipdir = d / "clips"
    wavs = sorted(clipdir.glob("*.wav")) if clipdir.exists() else []
    if not wavs:
        return
    stem = IDX_RE.sub("", wavs[0].name)  # e.g. "i-and-mine-and-namabhasa-stage"
    for r in rows:
        r["file"] = f"clips/{stem}_{r['idx']:04d}.wav"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/out/lecture1")
    ap.add_argument("--force", action="store_true", help="re-render thumbnails that already exist")
    args = ap.parse_args()

    d = Path(args.dir).resolve()
    if not (d / "qa_asr_report.tsv").exists():
        sys.exit(f"no qa_asr_report.tsv in {d}")

    rows = load_records(d)
    fix_filenames(d, rows)
    print(f"[gallery] {len(rows)} clips from {d.name}")

    # render thumbnails
    thumbs = d / "_thumbs"
    done = 0
    for r in rows:
        png = d / r["thumb"]
        wav = d / r["file"]
        if png.exists() and not args.force:
            continue
        if not wav.exists():
            print(f"  ! missing wav {wav.name}")
            continue
        render_thumb(wav, png)
        done += 1
        if done % 20 == 0:
            print(f"  rendered {done} ...")
    print(f"[gallery] thumbnails: {done} rendered, {len(rows)-done} reused")

    # emit html
    html = HTML_TEMPLATE.replace("__TITLE__", d.name)
    html = html.replace("__DATA__", json.dumps(rows, ensure_ascii=False))
    out = d / "qa_gallery.html"
    out.write_text(html, encoding="utf-8")
    print(f"[gallery] wrote {out}")
    print(f"[gallery] open it:  start {out}")


# =====================================================================================
# The gallery (vanilla HTML/CSS/JS). Defaults each clip to the gate's verdict (PASS->keep,
# FAIL->cut); you only press K/X where you DISAGREE. Reviewed/overrides tracked; export -> TSV.
# =====================================================================================
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA gallery — __TITLE__</title>
<style>
  :root{ --bg:#0d1117; --panel:#161b22; --line:#222c38; --fg:#e6edf3; --mut:#8b949e;
         --keep:#2ea043; --cut:#e5534b; --warn:#d29922; --acc:#5cc8ff; }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif; }
  header{ display:flex; align-items:center; gap:16px; padding:10px 16px; background:var(--panel);
          border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; flex-wrap:wrap; }
  header b{ font-size:15px; } .sp{ flex:1; }
  .pill{ padding:2px 9px; border-radius:99px; font-size:12px; font-weight:600; }
  .pill.keep{ background:rgba(46,160,67,.18); color:#56d364; }
  .pill.cut{ background:rgba(229,83,75,.18); color:#ff7b72; }
  .pill.rev{ background:rgba(92,200,255,.16); color:var(--acc); }
  .pill.ovr{ background:rgba(210,153,34,.18); color:#e3b341; }
  button,select{ background:#21262d; color:var(--fg); border:1px solid var(--line); border-radius:6px;
          padding:5px 10px; font:inherit; cursor:pointer; }
  button:hover{ border-color:#3d4756; }
  .wrap{ display:grid; grid-template-columns:230px 1fr; height:calc(100vh - 49px); }
  .strip{ overflow-y:auto; border-right:1px solid var(--line); background:#0b0f14; }
  .row{ display:flex; align-items:center; gap:8px; padding:6px 10px; border-left:3px solid transparent;
        cursor:pointer; font-size:12.5px; color:var(--mut); white-space:nowrap; }
  .row:hover{ background:#11161d; }
  .row.cur{ background:#161d27; color:var(--fg); border-left-color:var(--acc); }
  .row .dot{ width:8px; height:8px; border-radius:99px; flex:none; }
  .row.keep .dot{ background:var(--keep); } .row.cut .dot{ background:var(--cut); }
  .row.unrev{ opacity:.55; } .row.ovr{ font-weight:600; }
  .row . org{ margin-left:auto; font-size:10px; color:var(--warn); }
  .main{ overflow-y:auto; padding:18px 22px; }
  .head{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .head h2{ margin:0; font-size:20px; }
  .badge{ padding:2px 8px; border-radius:5px; font-size:12px; font-weight:700; }
  .badge.PASS{ background:rgba(46,160,67,.16); color:#56d364; }
  .badge.FAIL{ background:rgba(229,83,75,.16); color:#ff7b72; }
  .specwrap{ position:relative; width:100%; max-width:900px; cursor:crosshair; line-height:0;
             outline:1px solid var(--line); border-radius:8px 8px 0 0; overflow:hidden; }
  img.spec{ width:100%; display:block; background:#0d1117; }
  .cur{ position:absolute; top:0; bottom:0; width:0; border-left:1px solid; pointer-events:none; }
  .cur.hov{ border-color:rgba(255,255,255,.5); display:none; }
  .cur.play{ border-color:var(--acc); box-shadow:0 0 7px var(--acc); display:none; }
  .tip{ position:absolute; top:3px; transform:translateX(-50%); background:#000d; color:#fff;
        font:11px ui-monospace,monospace; padding:1px 5px; border-radius:4px; pointer-events:none;
        display:none; white-space:nowrap; }
  .ruler{ position:relative; height:21px; width:100%; max-width:900px; background:var(--panel);
          outline:1px solid var(--line); border-radius:0 0 8px 8px; }
  .ruler .tk{ position:absolute; top:0; height:4px; width:1px; background:#39424e; }
  .ruler .tk.maj{ height:8px; background:#6b7785; }
  .ruler .lb{ position:absolute; top:8px; transform:translateX(-50%); font:10px ui-monospace,monospace;
              color:var(--mut); white-space:nowrap; }
  .tnow{ max-width:900px; font:13px ui-monospace,monospace; color:var(--fg); margin:6px 0 12px; }
  .text{ max-width:860px; font-size:17px; line-height:1.6; margin:14px 0; padding:12px 14px;
         background:var(--panel); border-radius:8px; border:1px solid var(--line); }
  .metrics{ display:flex; flex-wrap:wrap; gap:8px 16px; max-width:860px; color:var(--mut); font-size:12.5px; margin-bottom:12px; }
  .metrics b{ color:var(--fg); font-variant-numeric:tabular-nums; }
  .metrics .bad{ color:#ff7b72; } .metrics .ok{ color:#56d364; }
  .reasons{ color:#e3b341; }
  audio{ width:100%; max-width:860px; margin:6px 0 14px; }
  .acts{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; max-width:860px; }
  .big{ font-size:15px; padding:9px 18px; font-weight:600; }
  .big.keep{ border-color:var(--keep); } .big.keep:hover{ background:rgba(46,160,67,.15); }
  .big.cut{ border-color:var(--cut); } .big.cut:hover{ background:rgba(229,83,75,.15); }
  textarea{ width:100%; max-width:860px; margin-top:12px; background:var(--panel); color:var(--fg);
            border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; resize:vertical; }
  kbd{ background:#21262d; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
       padding:0 5px; font:12px ui-monospace,monospace; color:var(--fg); }
  .legend{ max-width:860px; margin-top:18px; color:var(--mut); font-size:12px; display:flex; flex-wrap:wrap; gap:6px 14px; }
  .prog{ font-variant-numeric:tabular-nums; }
</style>
</head>
<body>
<header>
  <b>QA · __TITLE__</b>
  <span class="prog" id="prog"></span>
  <span class="sp"></span>
  <label style="color:var(--mut)">view
    <select id="filter">
      <option value="all">all</option>
      <option value="fail">gate-FAIL only</option>
      <option value="unrev">unreviewed</option>
      <option value="ovr">overrides</option>
    </select>
  </label>
  <label style="color:var(--mut)">speed
    <select id="speed"><option>1</option><option>1.25</option><option>1.5</option><option>2</option></select>
  </label>
  <button id="export">Export decisions ▾</button>
</header>

<div class="wrap">
  <div class="strip" id="strip"></div>
  <div class="main" id="main"></div>
</div>

<script>
const DATA = __DATA__;
const LSKEY = "qa-__TITLE__";

// default decision = gate verdict; you only flip disagreements.
const saved = JSON.parse(localStorage.getItem(LSKEY) || "{}");
DATA.forEach(c=>{
  const s = saved[c.file] || {};
  c.decision = s.decision || (c.status==="PASS" ? "keep" : "cut");
  c.reviewed = !!s.reviewed;
  c.note = s.note || "";
});

const gateOf = c => c.status==="PASS" ? "keep" : "cut";
const isOverride = c => c.decision !== gateOf(c);

function order(){
  // FAIL first, then within each group worst-align first (most suspicious gets your fresh ears)
  return [...DATA].sort((a,b)=>{
    if((a.status==="FAIL")!==(b.status==="FAIL")) return a.status==="FAIL"?-1:1;
    return a.align_min - b.align_min;
  });
}
function view(){
  const f = document.getElementById("filter").value;
  let v = order();
  if(f==="fail")  v = v.filter(c=>c.status==="FAIL");
  if(f==="unrev") v = v.filter(c=>!c.reviewed);
  if(f==="ovr")   v = v.filter(c=>isOverride(c));
  return v.length ? v : order();
}
let V = view();
let cur = 0;

function persist(){
  const o={}; DATA.forEach(c=>{ o[c.file]={decision:c.decision,reviewed:c.reviewed,note:c.note}; });
  localStorage.setItem(LSKEY, JSON.stringify(o));
}
function fmt(x,d=2){ return x.toFixed(d); }

function renderStrip(){
  const el = document.getElementById("strip");
  el.innerHTML = "";
  V.forEach((c,i)=>{
    const r = document.createElement("div");
    r.className = "row "+c.decision+(i===cur?" cur":"")+(c.reviewed?"":" unrev")+(isOverride(c)?" ovr":"");
    r.innerHTML = `<span class="dot"></span>#${String(c.idx).padStart(3,"0")}
      <span style="color:var(--mut)">${c.status}</span>${isOverride(c)?'<span class="org">⟳</span>':''}`;
    r.onclick = ()=>{ cur=i; render(); };
    el.appendChild(r);
  });
  const c = document.getElementById("strip").querySelector(".cur");
  if(c) c.scrollIntoView({block:"nearest"});
}

function renderProg(){
  const rev = DATA.filter(c=>c.reviewed).length;
  const keep = DATA.filter(c=>c.decision==="keep").length;
  const ovr = DATA.filter(c=>isOverride(c)).length;
  document.getElementById("prog").innerHTML =
    `reviewed <b>${rev}/${DATA.length}</b> · keep <span class="pill keep">${keep}</span>
     cut <span class="pill cut">${DATA.length-keep}</span> · overrides <span class="pill ovr">${ovr}</span>`;
}

function metric(label, val, bad){
  return `${label} <b class="${bad?'bad':''}">${val}</b>`;
}
function render(){
  const c = V[cur];
  renderStrip(); renderProg();
  const m = document.getElementById("main");
  m.innerHTML = `
    <div class="head">
      <h2>#${String(c.idx).padStart(3,"0")}</h2>
      <span class="badge ${c.status}">gate: ${c.status}</span>
      <span class="pill ${c.decision}">you: ${c.decision}</span>
      ${c.reviewed?'<span class="pill rev">reviewed</span>':''}
      ${isOverride(c)?'<span class="pill ovr">⟳ override</span>':''}
      <span class="sp"></span><span style="color:var(--mut)">${cur+1} / ${V.length} in view</span>
    </div>
    <div class="specwrap" id="specwrap" title="hover = read time · click = seek">
      <img class="spec" src="${c.thumb}" alt="spectrogram" id="spec">
      <div class="cur hov" id="hov"></div>
      <div class="cur play" id="play"></div>
      <div class="tip" id="tip"></div>
    </div>
    <div class="ruler" id="ruler"></div>
    <div class="tnow"><span id="tnow">0.000</span> / ${fmt(c.dur,2)}s
      &nbsp;·&nbsp;<span style="color:var(--mut)">waveform + mel (0–8&nbsp;kHz) · hover to read ms, click to seek</span></div>
    <div class="text">${c.text ? c.text.replace(/</g,"&lt;") : '<i style="color:var(--mut)">(no transcript)</i>'}</div>
    <div class="metrics">
      ${metric("dur", fmt(c.dur)+"s")}
      ${metric("lead-sil", fmt(c.lead)+"s", c.lead>0.6)}
      ${metric("trail-sil", fmt(c.trail)+"s", c.trail>1.0)}
      ${metric("align-min", fmt(c.align_min,1), c.align_min<-40)}
      ${metric("align-mean", fmt(c.align_mean,1))}
      ${metric("cer-core", fmt(c.cer_core), c.cer_core>0.85)}
      ${metric("wer", fmt(c.wer))}
      ${c.reasons?`<span class="reasons">⚑ ${c.reasons}</span>`:''}
    </div>
    <audio id="aud" src="${c.file}" controls preload="none"></audio>
    <div class="acts">
      <button class="big keep" onclick="decide('keep')">Keep <kbd>K</kbd></button>
      <button class="big cut" onclick="decide('cut')">Cut <kbd>X</kbd></button>
      <button onclick="replay()">Replay <kbd>R</kbd></button>
      <button onclick="step(-1)">‹ Prev <kbd>←</kbd></button>
      <button onclick="step(1)">Next › <kbd>→</kbd></button>
    </div>
    <textarea id="note" rows="2" placeholder="note (why cut / threshold to tune) — press N to focus">${c.note||""}</textarea>
    <div class="legend">
      <span><kbd>Space</kbd> play/pause</span><span><kbd>K</kbd> keep+next</span>
      <span><kbd>X</kbd> cut+next</span><span><kbd>R</kbd> replay</span>
      <span><kbd>←</kbd>/<kbd>→</kbd> prev/next</span><span><kbd>N</kbd> note</span>
      <span><kbd>1/2/3</kbd> speed</span><span><kbd>E</kbd> export</span>
      <span style="color:var(--mut)">default = gate verdict; press K/X only to confirm or flip.</span>
    </div>`;
  const aud = document.getElementById("aud");
  aud.playbackRate = parseFloat(document.getElementById("speed").value);
  wireSpec(c);
  document.getElementById("note").onchange = e=>{ c.note=e.target.value; persist(); };
}

// --- interactive spectrogram: pixel<->time is linear because the image is full-bleed (0..dur edge-to-edge)
function wireSpec(c){
  const wrap=document.getElementById("specwrap"), hov=document.getElementById("hov"),
        play=document.getElementById("play"), tip=document.getElementById("tip"),
        tnow=document.getElementById("tnow"), aud=document.getElementById("aud");
  const dur = c.dur || 1;
  const W = ()=> wrap.clientWidth;
  const x2t = x => Math.max(0, Math.min(1, x / W())) * dur;
  const t2x = t => (t / dur) * W();
  wrap.onmousemove = e=>{
    const x = e.clientX - wrap.getBoundingClientRect().left;
    hov.style.display="block"; hov.style.left=x+"px";
    tip.style.display="block"; tip.style.left=x+"px"; tip.textContent = x2t(x).toFixed(3)+"s";
  };
  wrap.onmouseleave = ()=>{ hov.style.display="none"; tip.style.display="none"; };
  wrap.onclick = e=>{
    aud.currentTime = x2t(e.clientX - wrap.getBoundingClientRect().left);
    aud.play().catch(()=>{});
  };
  aud.ontimeupdate = ()=>{
    play.style.display="block"; play.style.left = t2x(aud.currentTime)+"px";
    tnow.textContent = aud.currentTime.toFixed(3);
  };
  buildRuler(c);
  window.onresize = ()=>{ buildRuler(c); play.style.left = t2x(aud.currentTime)+"px"; };
}
function niceStep(dur){
  const want = dur/9;
  for(const s of [0.05,0.1,0.2,0.25,0.5,1,2,2.5,5,10,15,30,60]) if(s>=want) return s;
  return 60;
}
function buildRuler(c){
  const el=document.getElementById("ruler"); if(!el) return;
  const W=document.getElementById("specwrap").clientWidth, dur=c.dur||1, maj=niceStep(dur), min=maj/5;
  el.innerHTML=""; el.style.width=W+"px";
  for(let n=0; n*min<=dur+1e-6; n++){
    const t=n*min, isMaj=(n % 5===0);
    const tk=document.createElement("div"); tk.className="tk"+(isMaj?" maj":"");
    tk.style.left=((t/dur)*W)+"px"; el.appendChild(tk);
    if(isMaj){ const lb=document.createElement("div"); lb.className="lb";
      lb.style.left=((t/dur)*W)+"px"; lb.textContent=t.toFixed(maj<1?1:0)+"s"; el.appendChild(lb); }
  }
}

function decide(d){
  const c = V[cur];
  c.decision = d; c.reviewed = true; persist();
  step(1, true);
}
function step(dir, auto){
  if(dir>0 && cur>=V.length-1){ render(); if(auto) flashDone(); return; }
  cur = Math.max(0, Math.min(V.length-1, cur+dir));
  render();
  if(auto){ const a=document.getElementById("aud"); a.play().catch(()=>{}); }
}
function replay(){ const a=document.getElementById("aud"); a.currentTime=0; a.play().catch(()=>{}); }
function flashDone(){ document.getElementById("prog").insertAdjacentHTML("beforeend",
  ' <span class="pill rev">✓ end of view</span>'); }

document.getElementById("filter").onchange = ()=>{ V=view(); cur=0; render(); };
document.getElementById("speed").onchange = ()=>{ const a=document.getElementById("aud"); if(a) a.playbackRate=parseFloat(event.target.value); };

document.addEventListener("keydown", e=>{
  if(e.target.tagName==="TEXTAREA"){ if(e.key==="Escape") e.target.blur(); return; }
  if(e.target.tagName==="SELECT") return;
  const a=document.getElementById("aud");
  switch(e.key){
    case " ": e.preventDefault(); if(a) (a.paused?a.play():a.pause()); break;
    case "k": case "K": decide("keep"); break;
    case "x": case "X": decide("cut"); break;
    case "r": case "R": replay(); break;
    case "ArrowRight": e.preventDefault(); step(1); break;
    case "ArrowLeft": e.preventDefault(); step(-1); break;
    case "n": case "N": e.preventDefault(); document.getElementById("note").focus(); break;
    case "1": setSpeed("1"); break;
    case "2": setSpeed("1.25"); break;
    case "3": setSpeed("1.5"); break;
    case "e": case "E": doExport(); break;
  }
});
function setSpeed(v){ document.getElementById("speed").value=v; const a=document.getElementById("aud"); if(a) a.playbackRate=parseFloat(v); }

function doExport(){
  const hdr = ["idx","file","gate","decision","reviewed","override","lead","trail","align_min","cer_core","note","text"];
  const lines = [hdr.join("\t")];
  order().forEach(c=>{
    lines.push([c.idx,c.file,gateOf(c),c.decision,c.reviewed,isOverride(c),
      c.lead,c.trail,c.align_min,c.cer_core,(c.note||"").replace(/\t/g," "),(c.text||"").replace(/\t/g," ")].join("\t"));
  });
  // keep-only clean manifest (jsonl) for training
  const manifest = order().filter(c=>c.decision==="keep")
    .map(c=>JSON.stringify({audio:c.file,text:c.text,duration:c.dur})).join("\n");
  download("qa_decisions.tsv", lines.join("\n"));
  download("train.reviewed.jsonl", manifest);
  const ovr = DATA.filter(c=>isOverride(c));
  alert(`Exported.\n\nreviewed ${DATA.filter(c=>c.reviewed).length}/${DATA.length}\nkeeps ${DATA.filter(c=>c.decision==="keep").length}\noverrides vs gate: ${ovr.length}`);
}
function download(name, text){
  const b = new Blob([text], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href), 1000);
}
document.getElementById("export").onclick = doExport;

render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
