#!/usr/bin/env python
# =====================================================================================
# build_qa_gallery_ws.py — wavesurfer.js QA gallery (LIVE, zoomable waveform + mel spectrogram).
#
# Unlike build_qa_gallery.py (static PNG thumbnails) this renders the spectrogram CLIENT-SIDE from the
# actual audio, so you can zoom into any region (no raster ceiling) and every clip shows at the SAME
# px/sec scale (no stretch). It needs the page served over http (Chrome blocks the audio-decode that
# the spectrogram needs on file://):
#
#   python -m http.server 8000 --directory data/out/lecture1
#   -> open  http://localhost:8000/qa_gallery_ws.html  in Chrome/Edge   (NOT VSCode's preview: no audio)
#
# Build:  python scripts/build_qa_gallery_ws.py --dir data/out/lecture1
# (vendors wavesurfer v7 + spectrogram plugin into <dir>/vendor on first run; offline after that)
# =====================================================================================
import argparse, csv, json, re, sys, urllib.request
from pathlib import Path

IDX_RE = re.compile(r"_(\d+)\.wav$")
VENDOR = {
    "vendor/wavesurfer.min.js": "https://unpkg.com/wavesurfer.js@7.12.8/dist/wavesurfer.min.js",
    "vendor/plugins/spectrogram.min.js": "https://unpkg.com/wavesurfer.js@7.12.8/dist/plugins/spectrogram.min.js",
}


def clip_index(p):
    m = IDX_RE.search(p); return int(m.group(1)) if m else -1


def ensure_vendor(d: Path):
    for rel, url in VENDOR.items():
        p = d / rel
        if p.exists() and p.stat().st_size > 1000:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [vendor] downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())


def load_records(d: Path):
    texts = {}
    tj = d / "train.jsonl"
    if tj.exists():
        for line in tj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line); texts[clip_index(o["audio"])] = o.get("text", "")
    stem = "clip"
    wavs = sorted((d / "clips").glob("*.wav")) if (d / "clips").exists() else []
    if wavs:
        stem = IDX_RE.sub("", wavs[0].name)
    rows = []
    with open(d / "qa_asr_report.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["clip"])
            rows.append({
                "idx": i, "file": f"clips/{stem}_{i:04d}.wav", "text": texts.get(i, ""),
                "dur": float(r["dur"]), "align_min": float(r["align_min"]), "align_mean": float(r["align_mean"]),
                "cer_core": float(r["cer_core"]), "cer_raw": float(r["cer_raw"]), "wer": float(r["wer"]),
                "lead": float(r["lead"]), "trail": float(r["trail"]),
                "status": r["status"].strip(), "reasons": r["reasons"].strip(),
            })
    return rows, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/out/lecture1")
    ap.add_argument("--only", default="", help="comma/space-separated clip indices to include (default: all)")
    ap.add_argument("--out", default="qa_gallery_ws.html", help="output html filename (written in --dir)")
    ap.add_argument("--slug", default="", help="title + localStorage-key slug (default: dir name). Use a distinct slug "
                                               "for a subset so its keep/cut verdicts stay separate from the full gallery.")
    args = ap.parse_args()
    d = Path(args.dir).resolve()
    if not (d / "qa_asr_report.tsv").exists():
        sys.exit(f"no qa_asr_report.tsv in {d}")
    ensure_vendor(d)
    rows, stem = load_records(d)
    if args.only.strip():
        keep = {int(x) for x in re.split(r"[,\s]+", args.only.strip()) if x.isdigit()}
        rows = [r for r in rows if r["idx"] in keep]
    slug = args.slug.strip() or d.name
    print(f"[ws-gallery] {len(rows)} clips from {d.name} (slug={slug})")
    html = (HTML.replace("__TITLE__", slug)
                .replace("__SLUG__", slug)
                .replace("__DATA__", json.dumps(rows, ensure_ascii=False)))
    out = d / args.out
    out.write_text(html, encoding="utf-8")
    print(f"[ws-gallery] wrote {out}")
    print(f"[ws-gallery] serve it:\n    python -m http.server 8000 --directory {d}")
    print(f"    open http://localhost:8000/{args.out} in Chrome/Edge")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA (wavesurfer) — __TITLE__</title>
<script src="./vendor/wavesurfer.min.js"></script>
<script src="./vendor/plugins/spectrogram.min.js"></script>
<style>
  :root{ --bg:#0d1117; --panel:#161b22; --line:#222c38; --fg:#e6edf3; --mut:#8b949e;
         --keep:#2ea043; --cut:#e5534b; --warn:#d29922; --acc:#5cc8ff; }
  *{ box-sizing:border-box; } body{ margin:0; background:var(--bg); color:var(--fg);
     font:14px/1.5 ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif; }
  #filebanner{ display:none; background:#7d1a16; color:#fff; padding:8px 16px; font-weight:600; }
  #filebanner code{ background:#0006; padding:1px 6px; border-radius:4px; }
  header{ display:flex; align-items:center; gap:16px; padding:10px 16px; background:var(--panel);
          border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; flex-wrap:wrap; }
  header b{ font-size:15px; } .sp{ flex:1; }
  .pill{ padding:2px 9px; border-radius:99px; font-size:12px; font-weight:600; }
  .pill.keep{ background:rgba(46,160,67,.18); color:#56d364; }
  .pill.cut{ background:rgba(229,83,75,.18); color:#ff7b72; }
  .pill.rev{ background:rgba(92,200,255,.16); color:var(--acc); }
  .pill.ovr{ background:rgba(210,153,34,.18); color:#e3b341; }
  button,select{ background:#21262d; color:var(--fg); border:1px solid var(--line); border-radius:6px;
          padding:5px 10px; font:inherit; cursor:pointer; } button:hover{ border-color:#3d4756; }
  .wrap{ display:grid; grid-template-columns:230px 1fr; height:calc(100vh - 49px); }
  .strip{ overflow-y:auto; border-right:1px solid var(--line); background:#0b0f14; }
  .row{ display:flex; align-items:center; gap:8px; padding:6px 10px; border-left:3px solid transparent;
        cursor:pointer; font-size:12.5px; color:var(--mut); white-space:nowrap; }
  .row:hover{ background:#11161d; } .row.cur{ background:#161d27; color:var(--fg); border-left-color:var(--acc); }
  .row .dot{ width:8px; height:8px; border-radius:99px; flex:none; }
  .row.keep .dot{ background:var(--keep); } .row.cut .dot{ background:var(--cut); }
  .row.unrev{ opacity:.55; } .row.ovr{ font-weight:600; }
  .main{ overflow-y:auto; padding:16px 20px; }
  .head{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }
  .head h2{ margin:0; font-size:20px; }
  .badge{ padding:2px 8px; border-radius:5px; font-size:12px; font-weight:700; }
  .badge.PASS{ background:rgba(46,160,67,.16); color:#56d364; } .badge.FAIL{ background:rgba(229,83,75,.16); color:#ff7b72; }
  #wave{ background:#0d1117; border:1px solid var(--line); border-radius:8px; padding:6px; max-width:1100px; }
  .zoombar{ display:flex; align-items:center; gap:12px; max-width:1100px; margin:8px 0 4px; color:var(--mut); font-size:13px; }
  .zoombar input[type=range]{ width:230px; } #tnow{ font:13px ui-monospace,monospace; color:var(--fg); }
  .text{ max-width:1100px; font-size:17px; line-height:1.6; margin:12px 0; padding:12px 14px;
         background:var(--panel); border-radius:8px; border:1px solid var(--line); }
  .metrics{ display:flex; flex-wrap:wrap; gap:8px 16px; max-width:1100px; color:var(--mut); font-size:12.5px; margin-bottom:12px; }
  .metrics b{ color:var(--fg); font-variant-numeric:tabular-nums; } .metrics .bad{ color:#ff7b72; }
  .reasons{ color:#e3b341; }
  .acts{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; max-width:1100px; }
  .big{ font-size:15px; padding:9px 18px; font-weight:600; }
  .big.keep{ border-color:var(--keep); } .big.keep:hover{ background:rgba(46,160,67,.15); }
  .big.cut{ border-color:var(--cut); } .big.cut:hover{ background:rgba(229,83,75,.15); }
  textarea{ width:100%; max-width:1100px; margin-top:12px; background:var(--panel); color:var(--fg);
            border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; resize:vertical; }
  kbd{ background:#21262d; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
       padding:0 5px; font:12px ui-monospace,monospace; } .prog{ font-variant-numeric:tabular-nums; }
  .legend{ max-width:1100px; margin-top:16px; color:var(--mut); font-size:12px; display:flex; flex-wrap:wrap; gap:6px 14px; }
</style>
</head>
<body>
<div id="filebanner">⚠ Opened as a local file — the spectrogram needs a server. Run
  <code>python -m http.server 8000 --directory .</code> in this folder, then open
  <code>http://localhost:8000/qa_gallery_ws.html</code> in Chrome/Edge.</div>
<header>
  <b>QA · __TITLE__</b>
  <span class="prog" id="prog"></span>
  <span class="sp"></span>
  <label style="color:var(--mut)">view
    <select id="filter"><option value="all">all</option><option value="fail">gate-FAIL only</option>
      <option value="unrev">unreviewed</option><option value="ovr">overrides</option></select></label>
  <button id="export">Export decisions ▾</button>
</header>

<div class="wrap">
  <div class="strip" id="strip"></div>
  <div class="main">
    <div class="head" id="head"></div>
    <div id="wave"></div>
    <div class="zoombar">
      <span>zoom</span><input id="zoom" type="range" min="40" max="400" value="120">
      <span id="zlbl">120 px/s</span>
      <span id="tnow">0.000 / 0.00s</span>
      <span class="sp"></span>
      <span>speed</span><select id="speed"><option>1</option><option>1.25</option><option>1.5</option><option>2</option></select>
      <button onclick="ws&&ws.playPause()">Play/Pause <kbd>Space</kbd></button>
    </div>
    <div class="text" id="text"></div>
    <div class="metrics" id="metrics"></div>
    <div class="acts">
      <button class="big keep" onclick="decide('keep')">Keep <kbd>K</kbd></button>
      <button class="big cut" onclick="decide('cut')">Cut <kbd>X</kbd></button>
      <button onclick="replay()">Replay <kbd>R</kbd></button>
      <button onclick="step(-1)">‹ Prev <kbd>←</kbd></button>
      <button onclick="step(1)">Next › <kbd>→</kbd></button>
    </div>
    <textarea id="note" rows="2" placeholder="note (why cut / threshold to tune) — press N to focus"></textarea>
    <div class="legend">
      <span><kbd>Space</kbd> play/pause</span><span><kbd>K</kbd> keep+next</span><span><kbd>X</kbd> cut+next</span>
      <span><kbd>R</kbd> replay</span><span><kbd>←</kbd>/<kbd>→</kbd> prev/next</span><span><kbd>N</kbd> note</span>
      <span><kbd>+</kbd>/<kbd>-</kbd> zoom</span><span><kbd>1/2/3</kbd> speed</span><span><kbd>E</kbd> export</span>
      <span style="color:var(--mut)">drag on the waveform to zoom-scroll; click to seek; default = gate verdict.</span>
    </div>
  </div>
</div>

<script>
const DATA = __DATA__;
const LSKEY = "qaws-__SLUG__";
if (location.protocol === "file:") document.getElementById("filebanner").style.display = "block";

const saved = JSON.parse(localStorage.getItem(LSKEY) || "{}");
DATA.forEach(c=>{ const s=saved[c.file]||{};
  c.decision=s.decision||(c.status==="PASS"?"keep":"cut"); c.reviewed=!!s.reviewed; c.note=s.note||""; });
const gateOf=c=>c.status==="PASS"?"keep":"cut", isOverride=c=>c.decision!==gateOf(c);
function order(){ return [...DATA].sort((a,b)=>{
  if((a.status==="FAIL")!==(b.status==="FAIL")) return a.status==="FAIL"?-1:1; return a.align_min-b.align_min; }); }
function view(){ const f=document.getElementById("filter").value; let v=order();
  if(f==="fail")v=v.filter(c=>c.status==="FAIL"); if(f==="unrev")v=v.filter(c=>!c.reviewed);
  if(f==="ovr")v=v.filter(isOverride); return v.length?v:order(); }
let V=view(), cur=0, zoomLvl=120;
function persist(){ const o={}; DATA.forEach(c=>o[c.file]={decision:c.decision,reviewed:c.reviewed,note:c.note});
  localStorage.setItem(LSKEY,JSON.stringify(o)); }
function fmt(x,d=2){ return x.toFixed(d); }

// ---- wavesurfer: one instance, reused across clips --------------------------------
let ws=null, pendingPlay=false;
function initWS(){
  ws = WaveSurfer.create({ container:"#wave", height:90, waveColor:"#5cc8ff", progressColor:"#2b7bb0",
    cursorColor:"#fff", cursorWidth:2, minPxPerSec:zoomLvl, autoScroll:true, interact:true, sampleRate:16000 });
  // noverlap is set EXPLICITLY (not auto): hop = fftSamples - noverlap = 64 samples -> ~250 columns/sec,
  // a fixed high time-resolution independent of zoom, so moderate zoom stays crisp (the slider is capped
  // to the crisp range below). Without this, wavesurfer ties columns to render-width and blurs on zoom.
  ws.registerPlugin(WaveSurfer.Spectrogram.create({ labels:true, height:256, scale:"mel",
    fftSamples:1024, noverlap:960, windowFunc:"hann", frequencyMin:0, frequencyMax:8000,
    labelsBackground:"rgba(0,0,0,.4)" }));
  ws.on("ready", ()=>{ try{ ws.zoom(zoomLvl); }catch(e){}
    ws.setPlaybackRate(parseFloat(document.getElementById("speed").value));
    if(pendingPlay){ ws.play().catch(()=>{}); pendingPlay=false; } });
  ws.on("timeupdate", t=>{ document.getElementById("tnow").textContent =
    t.toFixed(3)+" / "+(ws.getDuration()||0).toFixed(2)+"s"; });
}
function loadClip(c){ document.getElementById("tnow").textContent="0.000 / "+fmt(c.dur)+"s"; ws.load(c.file); }

function renderStrip(){ const el=document.getElementById("strip"); el.innerHTML="";
  V.forEach((c,i)=>{ const r=document.createElement("div");
    r.className="row "+c.decision+(i===cur?" cur":"")+(c.reviewed?"":" unrev")+(isOverride(c)?" ovr":"");
    r.innerHTML=`<span class="dot"></span>#${String(c.idx).padStart(3,"0")} <span style="color:var(--mut)">${c.status}</span>${isOverride(c)?" ⟳":""}`;
    r.onclick=()=>{ cur=i; render(); }; el.appendChild(r); });
  const c=document.querySelector(".row.cur"); if(c) c.scrollIntoView({block:"nearest"}); }
function renderProg(){ const rev=DATA.filter(c=>c.reviewed).length, keep=DATA.filter(c=>c.decision==="keep").length,
  ovr=DATA.filter(isOverride).length;
  document.getElementById("prog").innerHTML=`reviewed <b>${rev}/${DATA.length}</b> · keep <span class="pill keep">${keep}</span> cut <span class="pill cut">${DATA.length-keep}</span> · overrides <span class="pill ovr">${ovr}</span>`; }
function metric(l,v,bad){ return `${l} <b class="${bad?'bad':''}">${v}</b>`; }
function render(){ const c=V[cur]; renderStrip(); renderProg();
  document.getElementById("head").innerHTML=`<h2>#${String(c.idx).padStart(3,"0")}</h2>
    <span class="badge ${c.status}">gate: ${c.status}</span><span class="pill ${c.decision}">you: ${c.decision}</span>
    ${c.reviewed?'<span class="pill rev">reviewed</span>':''}${isOverride(c)?'<span class="pill ovr">⟳ override</span>':''}
    <span class="sp"></span><span style="color:var(--mut)">${cur+1} / ${V.length} in view</span>`;
  document.getElementById("text").innerHTML = c.text ? c.text.replace(/</g,"&lt;") : '<i style="color:var(--mut)">(no transcript)</i>';
  document.getElementById("metrics").innerHTML =
    metric("dur",fmt(c.dur)+"s")+metric("lead-sil",fmt(c.lead)+"s",c.lead>0.6)+metric("trail-sil",fmt(c.trail)+"s",c.trail>1.0)+
    metric("align-min",fmt(c.align_min,1),c.align_min<-40)+metric("align-mean",fmt(c.align_mean,1))+
    metric("cer-core",fmt(c.cer_core),c.cer_core>0.85)+metric("wer",fmt(c.wer))+
    (c.reasons?`<span class="reasons">⚑ ${c.reasons}</span>`:"");
  document.getElementById("note").value = c.note||"";
  loadClip(c);
}
function decide(d){ const c=V[cur]; c.decision=d; c.reviewed=true; persist(); step(1,true); }
function step(dir,auto){ if(dir>0&&cur>=V.length-1){ render(); return; }
  cur=Math.max(0,Math.min(V.length-1,cur+dir)); pendingPlay=!!auto; render(); }
function replay(){ if(ws){ ws.setTime(0); ws.play().catch(()=>{}); } }
function setZoom(v){ zoomLvl=Math.max(40,Math.min(400,v));
  document.getElementById("zoom").value=zoomLvl; document.getElementById("zlbl").textContent=zoomLvl+" px/s";
  if(ws){ try{ ws.zoom(zoomLvl); }catch(e){} } }
function setSpeed(v){ document.getElementById("speed").value=v; if(ws) ws.setPlaybackRate(parseFloat(v)); }

document.getElementById("filter").onchange=()=>{ V=view(); cur=0; render(); };
document.getElementById("zoom").oninput=e=>setZoom(Number(e.target.value));
document.getElementById("speed").onchange=e=>setSpeed(e.target.value);
document.getElementById("note").onchange=e=>{ V[cur].note=e.target.value; persist(); };

document.addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA"){ if(e.key==="Escape")e.target.blur(); return; }
  if(e.target.tagName==="SELECT"||e.target.tagName==="INPUT") return;
  switch(e.key){
    case " ": e.preventDefault(); if(ws)ws.playPause(); break;
    case "k": case "K": decide("keep"); break;
    case "x": case "X": decide("cut"); break;
    case "r": case "R": replay(); break;
    case "ArrowRight": e.preventDefault(); step(1); break;
    case "ArrowLeft": e.preventDefault(); step(-1); break;
    case "n": case "N": e.preventDefault(); document.getElementById("note").focus(); break;
    case "+": case "=": setZoom(zoomLvl+40); break;
    case "-": case "_": setZoom(zoomLvl-40); break;
    case "1": setSpeed("1"); break; case "2": setSpeed("1.25"); break; case "3": setSpeed("1.5"); break;
    case "e": case "E": doExport(); break;
  }
});

function doExport(){
  const hdr=["idx","file","gate","decision","reviewed","override","lead","trail","align_min","cer_core","note","text"];
  const lines=[hdr.join("\t")];
  order().forEach(c=>lines.push([c.idx,c.file,gateOf(c),c.decision,c.reviewed,isOverride(c),
    c.lead,c.trail,c.align_min,c.cer_core,(c.note||"").replace(/[\t\r\n]+/g," "),(c.text||"").replace(/[\t\r\n]+/g," ")].join("\t")));
  const manifest=order().filter(c=>c.decision==="keep").map(c=>JSON.stringify({audio:c.file,text:c.text,duration:c.dur})).join("\n");
  download("qa_decisions.tsv",lines.join("\n")); download("train.reviewed.jsonl",manifest);
  alert(`Exported.\n\nreviewed ${DATA.filter(c=>c.reviewed).length}/${DATA.length}\nkeeps ${DATA.filter(c=>c.decision==="keep").length}\noverrides vs gate: ${DATA.filter(isOverride).length}`);
}
function download(name,text){ const b=new Blob([text],{type:"text/plain;charset=utf-8"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(b); a.download=name; a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000); }
document.getElementById("export").onclick=doExport;

initWS(); render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
