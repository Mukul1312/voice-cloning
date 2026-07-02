"""
build_hi_review.py — transcript-review gallery for the Hindi clip set.

Reads the Whisper-draft manifests (data/hi/lectures/<slug>/train_hi.jsonl) + their clips and emits a
static page (voice_lab/hi_review/) where you:
  - play each clip and EDIT its Devanagari transcript,
  - tick "drop" for junk / non-Hindi clips,
  - tick "⭐ref" to mark a clip as a probe reference,
then Download corrected JSONL (train_hi_corrected.jsonl) and/or Download probe refs (probe_refs.jsonl,
= the starred clips with their corrected text). Edits autosave (localStorage), so a long review survives
reloads and rebuilds (same clip ids -> same keys).

Serve with:  python -m http.server 8000 --directory voice_lab   ->   http://localhost:8000/hi_review/
  python scripts/build_hi_review.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HI = ROOT / "data" / "hi" / "lectures"
OUT = ROOT / "voice_lab" / "hi_review"
AUD = OUT / "audio"

HTML = r"""<!doctype html><html lang="hi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hindi transcript review — GGSM</title>
<style>
 :root{--bg:#12100f;--card:#1d1a18;--line:#332e2a;--ink:#f3eee9;--mut:#a89f97;--gold:#c9a24a;--drop:#b4574d}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
 .bar{position:sticky;top:0;background:#0d0b0a;border-bottom:1px solid var(--line);padding:10px 16px;
   display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
 .bar b{color:var(--gold)} .bar .sp{flex:1}
 button{cursor:pointer;border:1px solid var(--line);background:#241f1c;color:var(--ink);
   border-radius:8px;padding:8px 14px;font-size:14px}
 button.dl{background:var(--gold);color:#12100f;border-color:var(--gold);font-weight:600}
 .wrap{max-width:900px;margin:0 auto;padding:16px}
 h2{color:var(--mut);font-size:12px;letter-spacing:.14em;text-transform:uppercase;margin:22px 0 8px}
 .row{display:flex;gap:12px;align-items:flex-start;background:var(--card);border:1px solid var(--line);
   border-radius:10px;padding:10px 12px;margin:0 0 10px}
 .row.drop{opacity:.45;border-color:var(--drop)}
 .row.isref{border-color:var(--gold);box-shadow:inset 3px 0 0 var(--gold)}
 .n{flex:0 0 42px;color:var(--mut);font-size:12px;padding-top:6px}
 .mid{flex:1;min-width:0}
 audio{width:100%;height:34px;margin:0 0 6px}
 textarea{width:100%;min-height:46px;background:#15120f;color:var(--ink);border:1px solid var(--line);
   border-radius:6px;padding:8px;font:16px/1.5 'Noto Sans Devanagari',system-ui,serif;resize:vertical}
 .meta{color:var(--mut);font-size:11px;margin-top:4px}
 .rt{flex:0 0 78px;display:flex;flex-direction:column;gap:6px;align-items:flex-end;padding-top:4px}
 .rt label{font-size:12px;color:var(--drop);display:flex;gap:5px;align-items:center;cursor:pointer}
 .rt label.rf{color:var(--gold)}
 .done textarea{border-color:#4a7a4a}
</style></head><body>
<div class="bar">
 <b>Hindi transcript review</b>
 <span id="stat"></span>
 <span class="sp"></span>
 <button onclick="resetAll()">clear saved edits</button>
 <button onclick="downloadRefs()">⭐ Download probe refs</button>
 <button class="dl" onclick="download_()">⬇ Download corrected JSONL</button>
</div>
<div class="wrap" id="app"></div>
<script>
const D = __DATA__;
const KEY = "hi_review_v1";
let saved = {};
try{ saved = JSON.parse(localStorage.getItem(KEY) || "{}"); }catch(e){}

function persist(){ localStorage.setItem(KEY, JSON.stringify(saved)); stat(); }
function stat(){
  const n=D.length, ed=Object.values(saved).filter(s=>s&&s.text!==undefined).length,
        dr=Object.values(saved).filter(s=>s&&s.drop).length,
        rf=Object.values(saved).filter(s=>s&&s.ref).length;
  document.getElementById("stat").textContent = `${n} clips · ${ed} edited · ${dr} dropped · ${rf} ⭐ref`;
}
function resetAll(){ if(confirm("Clear all saved edits/drops/refs?")){ saved={}; persist(); render(); } }

function render(){
  const app=document.getElementById("app"); app.innerHTML="";
  let cur=null;
  D.forEach((x,i)=>{
    if(x.lecture!==cur){ cur=x.lecture; const h=document.createElement("h2"); h.textContent=cur; app.appendChild(h); }
    const s = saved[x.id] || {};
    const row=document.createElement("div");
    row.className="row"+(s.drop?" drop":"")+(s.ref?" isref":"")+(s.text!==undefined?" done":"");
    row.innerHTML =
      `<div class="n">#${i+1}</div>
       <div class="mid">
         <audio controls preload="none" src="${x.aud}"></audio>
         <textarea spellcheck="false">${(s.text!==undefined?s.text:x.text)||""}</textarea>
         <div class="meta">${x.id} · ${x.dur}s · @${x.start}s</div>
       </div>
       <div class="rt">
         <label class="rf"><input type="checkbox" class="refcb" ${s.ref?"checked":""}> ⭐ref</label>
         <label><input type="checkbox" class="dropcb" ${s.drop?"checked":""}> drop</label>
       </div>`;
    const ta=row.querySelector("textarea"),
          dcb=row.querySelector("input.dropcb"), rcb=row.querySelector("input.refcb");
    ta.oninput=()=>{ saved[x.id]={...(saved[x.id]||{}),text:ta.value}; row.classList.add("done"); persist(); };
    dcb.onchange=()=>{ saved[x.id]={...(saved[x.id]||{}),drop:dcb.checked}; row.classList.toggle("drop",dcb.checked); persist(); };
    rcb.onchange=()=>{ saved[x.id]={...(saved[x.id]||{}),ref:rcb.checked}; row.classList.toggle("isref",rcb.checked); persist(); };
    app.appendChild(row);
  });
  stat();
}
function _lines(pred){
  const out=[];
  for(const x of D){ const s=saved[x.id]||{}; if(!pred(s)) continue;
    out.push(JSON.stringify({...x.orig, text:(s.text!==undefined?s.text:x.text)})); }
  return out;
}
function _save(lines, name){
  const blob=new Blob([lines.join("\n")+"\n"],{type:"application/json"});
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name; a.click();
}
function download_(){ _save(_lines(s=>!s.drop), "train_hi_corrected.jsonl"); }
function downloadRefs(){
  const lines=_lines(s=>s.ref);
  if(!lines.length){ alert("No clips starred ⭐ref yet — tick a few clean ones for the probe."); return; }
  _save(lines, "probe_refs.jsonl");
}
render();
</script></body></html>"""


def main():
    rows = []
    for mani in sorted(HI.glob("*/train_hi.jsonl")):
        slug = mani.parent.name
        for line in mani.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            src = mani.parent / r["audio"]
            if not src.exists():
                continue
            rows.append((slug, r, src))
    if not rows:
        sys.exit("no clips found — pull data/hi/lectures/*/clips_hi + train_hi.jsonl from the pod first")

    AUD.mkdir(parents=True, exist_ok=True)
    for old in AUD.glob("*.wav"):
        old.unlink()
    items = []
    for (slug, r, src) in rows:
        aud_name = f"{slug}__{Path(r['audio']).name}"
        shutil.copy(src, AUD / aud_name)
        items.append({
            "id": aud_name, "lecture": slug, "aud": f"audio/{aud_name}",
            "text": r.get("text", ""),
            "dur": round(r.get("end", 0) - r.get("start", 0), 1), "start": r.get("start"),
            "orig": {k: r[k] for k in ("audio", "text", "lecture", "start", "end") if k in r},
        })
    (OUT / "index.html").write_text(HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False)),
                                    encoding="utf-8")
    print(f"wrote {OUT / 'index.html'}  ({len(items)} clips, audio copied to {AUD})")
    print("serve: python -m http.server 8000 --directory voice_lab  ->  http://localhost:8000/hi_review/")


if __name__ == "__main__":
    main()
