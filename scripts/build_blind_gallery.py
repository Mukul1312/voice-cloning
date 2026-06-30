"""
build_blind_gallery.py — a presentable, BLIND A/B listening test for picking the best
VoxCPM2 LoRA checkpoint by ear.

For each test sentence we have 5 candidates: the base model (LoRA disabled) + 4 LoRA
checkpoints (steps 150/250/350/500). This script:
  1. copies the 15 distinct wavs to OPAQUE names (cNN.wav) in samples/blind/ so the
     filename leaks nothing,
  2. shuffles the 5 candidates into slots A-E per test (independent shuffle per test),
  3. base64-encodes the slot->identity answer key so it can't be read from page source,
  4. emits a self-contained HTML gallery: ear-anchor (real GGS clip, labelled) + 5 blind
     players per test, 1-5 "sounds like him" rating + quick tags + notes, localStorage
     persistence, a Reveal button that scores each checkpoint, and JSON export.

Usage:  python scripts/build_blind_gallery.py
Open:   data/out/lecture1/samples/blind_gallery.html   (double-click)
"""
import base64
import json
import os
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "out" / "lecture1" / "samples"
BLIND = SAMPLES / "blind"
REF_SRC = ROOT / "data" / "out" / "lecture1" / "final" / "clips" / "ggs_l1_0112.wav"
LSKEY = "blind-lecture1-v1"
# Random, unprinted seed: each build is a genuinely fresh blind arrangement that nobody
# (author included) has seen. The reveal mapping is baked into the HTML, so the gallery
# is fully self-contained regardless. Pass an int as argv[1] to reproduce a build.
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else int.from_bytes(os.urandom(4), "big")

# Each test: a sentence (the words are NOT secret), a "condition" note, and the 5 candidate
# source files keyed by identity. 'base' = LoRA disabled (control).
TESTS = [
    {
        "title": "Test 1 — plain (voice baked into the LoRA, no prompt)",
        "text": "Real intelligence is to give up this illusion and inquire about Krishna.",
        "cond": "No reference clip given — any resemblance comes purely from the fine-tune.",
        "files": {
            "base": "s150_plain1_lora_disabled.wav",
            "150": "s150_plain1_with_lora.wav",
            "250": "s250_plain1_with_lora.wav",
            "350": "s350_plain1_with_lora.wav",
            "500": "s500_plain1_with_lora.wav",
        },
    },
    {
        "title": "Test 2 — plain, a different sentence",
        "text": "Chant the holy name with attention and devotion, and the Lord will reveal Himself.",
        "cond": "No reference clip given. A second sentence guards against one lucky line.",
        "files": {
            "base": "s150_plain2_lora_disabled.wav",
            "150": "s150_plain2_with_lora.wav",
            "250": "s250_plain2_with_lora.wav",
            "350": "s350_plain2_with_lora.wav",
            "500": "s500_plain2_with_lora.wav",
        },
    },
    {
        "title": "Test 3 — prompted (his real clip given as a voice prompt)",
        "text": "Real intelligence is to give up this illusion and inquire about Krishna.",
        "cond": "Same words as Test 1, but every candidate is conditioned on his real clip. "
                "Here even the base does zero-shot cloning — so this asks: does the LoRA beat "
                "plain prompt-based cloning?",
        "files": {
            "base": "s150_clone1_lora_disabled.wav",
            "150": "s150_clone1_with_lora.wav",
            "250": "s250_clone1_with_lora.wav",
            "350": "s350_clone1_with_lora.wav",
            "500": "s500_clone1_with_lora.wav",
        },
    },
]

SLOTS = ["A", "B", "C", "D", "E"]


def main():
    rng = random.Random(SEED)
    if BLIND.exists():
        shutil.rmtree(BLIND)
    BLIND.mkdir(parents=True)

    # 1) collect the distinct files, give them opaque names in shuffled order
    distinct = []
    for t in TESTS:
        for ident, fn in t["files"].items():
            if fn not in distinct:
                distinct.append(fn)
    opaque_order = list(range(1, len(distinct) + 1))
    rng.shuffle(opaque_order)
    opaque = {}  # src filename -> opaque relative path
    for fn, n in zip(distinct, opaque_order):
        oname = f"c{n:02d}.wav"
        shutil.copyfile(SAMPLES / fn, BLIND / oname)
        opaque[fn] = f"blind/{oname}"

    # reference (real him) — labelled, not blind
    ref_rel = ""
    if REF_SRC.exists():
        shutil.copyfile(REF_SRC, BLIND / "ref_him.wav")
        ref_rel = "blind/ref_him.wav"

    # 2) per test: shuffle identities into slots A-E; 3) base64 the answer key
    tests_js = []
    answer_log = []
    for ti, t in enumerate(TESTS):
        idents = list(t["files"].keys())
        rng.shuffle(idents)
        cands, truth = [], {}
        for slot, ident in zip(SLOTS, idents):
            cands.append({"slot": slot, "file": opaque[t["files"][ident]]})
            truth[slot] = ident
        truth_b64 = base64.b64encode(json.dumps(truth).encode()).decode()
        tests_js.append({
            "title": t["title"], "text": t["text"], "cond": t["cond"],
            "ref": ref_rel, "candidates": cands, "truth_b64": truth_b64,
        })
        answer_log.append({"test": ti + 1, **{c["slot"]: truth[c["slot"]] for c in cands}})

    html = TEMPLATE.replace("/*__TESTS__*/", json.dumps(tests_js, indent=2)) \
                   .replace("__LSKEY__", LSKEY)
    out = SAMPLES / "blind_gallery.html"
    out.write_text(html, encoding="utf-8")

    # write the answer key to a sibling file (for records; the UI never reads it)
    (SAMPLES / "blind_answer_key.json").write_text(
        json.dumps(answer_log, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    print(f"  opaque clips: {len(distinct)} -> {BLIND}")
    print(f"  reference clip: {'ref_him.wav' if ref_rel else 'MISSING'}")
    print(f"  answer key (do not peek): {SAMPLES / 'blind_answer_key.json'}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GGS clone — blind checkpoint test</title>
<style>
  :root { --bg:#0f1115; --card:#1a1d24; --line:#2a2e38; --ink:#e7e9ee; --mut:#9aa3b2;
          --accent:#7c9cff; --good:#46c08a; --warn:#e2b04a; --bad:#e2685a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:860px; margin:0 auto; padding:24px 18px 140px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--mut); margin:0 0 18px; }
  details { background:var(--card); border:1px solid var(--line); border-radius:10px;
            padding:10px 14px; margin:0 0 18px; }
  summary { cursor:pointer; font-weight:600; }
  details ul { margin:8px 0 0; padding-left:18px; color:var(--mut); }
  .test { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:18px; margin:0 0 22px; }
  .ttitle { font-weight:700; font-size:16px; }
  .cond { color:var(--mut); font-size:13px; margin:4px 0 2px; }
  .text { background:#11141b; border:1px solid var(--line); border-radius:8px;
          padding:10px 12px; margin:10px 0 14px; font-style:italic; }
  .ref { display:flex; align-items:center; gap:10px; margin:0 0 16px;
         padding:8px 10px; border:1px dashed var(--good); border-radius:8px; }
  .ref .lbl { color:var(--good); font-weight:600; font-size:13px; }
  .card { border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:0 0 10px; }
  .card.rated { border-color:var(--accent); }
  .crow { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .slot { font-weight:800; font-size:18px; width:26px; }
  .reveal-id { font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px;
               background:#2a2e38; color:var(--ink); display:none; }
  audio { height:34px; }
  .stars { display:flex; gap:4px; }
  .star { cursor:pointer; border:1px solid var(--line); background:#11141b; color:var(--mut);
          border-radius:6px; padding:3px 9px; font-size:13px; user-select:none; }
  .star.on { background:var(--accent); color:#0f1115; border-color:var(--accent); font-weight:700; }
  .tags { display:flex; gap:6px; flex-wrap:wrap; margin:10px 0 0; }
  .tag { cursor:pointer; border:1px solid var(--line); background:#11141b; color:var(--mut);
         border-radius:20px; padding:3px 10px; font-size:12px; user-select:none; }
  .tag.on { background:#2a2e38; color:var(--ink); border-color:var(--accent); }
  .note { width:100%; margin:10px 0 0; background:#11141b; color:var(--ink);
          border:1px solid var(--line); border-radius:8px; padding:7px 10px; font:inherit; font-size:13px; }
  .bar { position:fixed; left:0; right:0; bottom:0; background:#0c0e12cc; backdrop-filter:blur(6px);
         border-top:1px solid var(--line); padding:12px 18px; }
  .bar .inner { max-width:860px; margin:0 auto; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
  .prog { color:var(--mut); font-size:13px; }
  button.act { border:1px solid var(--line); background:var(--card); color:var(--ink);
               border-radius:8px; padding:8px 14px; cursor:pointer; font:inherit; font-weight:600; }
  button.act.primary { background:var(--accent); color:#0f1115; border-color:var(--accent); }
  button.act:disabled { opacity:.45; cursor:not-allowed; }
  #summary { background:var(--card); border:1px solid var(--accent); border-radius:14px;
             padding:18px; margin:0 0 22px; display:none; }
  table { width:100%; border-collapse:collapse; margin-top:10px; }
  th,td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); font-size:14px; }
  .win { color:var(--good); font-weight:700; }
  .pill { font-size:12px; padding:1px 7px; border-radius:20px; background:#2a2e38; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🎧 Blind checkpoint test — Gour Govinda Swami clone</h1>
  <p class="sub">Rate each candidate purely by ear. You don't know which is which until you press <b>Reveal</b>.</p>

  <details>
    <summary>How to do this (30 sec read)</summary>
    <ul>
      <li>Each test plays one sentence. First, the <b style="color:var(--good)">green clip is really him</b> — listen to it to calibrate your ear.</li>
      <li>Then rate candidates <b>A–E</b>: <b>1 = a stranger</b> &nbsp;…&nbsp; <b>5 = that's him</b>. Judge the <i>voice</i> (timbre, accent, cadence), not the words.</li>
      <li>Tag any defects you hear. Add a note if you like.</li>
      <li>One of the five in every test is the <b>base model (no fine-tune)</b> — the hidden control. If it scores high, the fine-tune isn't doing much.</li>
      <li>When all are rated, hit <b>Reveal &amp; score</b> — it averages each checkpoint across the three tests and names a winner. Then <b>Export</b>.</li>
    </ul>
  </details>

  <div id="summary"></div>
  <div id="tests"></div>
</div>

<div class="bar"><div class="inner">
  <span class="prog" id="prog">0 / 0 rated</span>
  <span style="flex:1"></span>
  <button class="act" id="resetBtn">Reset</button>
  <button class="act" id="revealBtn">🔓 Reveal &amp; score</button>
  <button class="act primary" id="exportBtn">⬇ Export</button>
</div></div>

<script>
const TESTS = /*__TESTS__*/;
const LSKEY = "__LSKEY__";
const TAGS = [
  ["him","🎯 sounds like him"], ["cutoff","✂️ cut off early"],
  ["robotic","🤖 robotic / buzzy"], ["words","❓ wrong / garbled words"],
  ["pace","🐌 unnatural pace"]
];
const IDS = ["base","150","250","350","500"];
const IDLABEL = {base:"BASE (no LoRA)", "150":"step 150", "250":"step 250", "350":"step 350", "500":"step 500"};

let state = load();
function load(){ try { return JSON.parse(localStorage.getItem(LSKEY)) || blank(); } catch(e){ return blank(); } }
function blank(){ return { ratings:{}, tags:{}, notes:{}, revealed:false }; }
function save(){ localStorage.setItem(LSKEY, JSON.stringify(state)); }
function key(ti, slot){ return "t"+ti+"_"+slot; }

function render(){
  const root = document.getElementById("tests");
  root.innerHTML = "";
  TESTS.forEach((t, ti) => {
    const sec = document.createElement("div");
    sec.className = "test";
    let h = `<div class="ttitle">${t.title}</div>
             <div class="cond">${t.cond}</div>
             <div class="text">“${t.text}”</div>`;
    if (t.ref) h += `<div class="ref"><span class="lbl">▶ Really him (anchor)</span>
                     <audio controls preload="none" src="${t.ref}"></audio></div>`;
    t.candidates.forEach(c => {
      const k = key(ti, c.slot);
      const r = state.ratings[k] || 0;
      const tg = state.tags[k] || [];
      const stars = [1,2,3,4,5].map(v =>
        `<span class="star ${r>=v?'on':''}" data-k="${k}" data-v="${v}">${v}</span>`).join("");
      const tags = TAGS.map(([id,lbl]) =>
        `<span class="tag ${tg.includes(id)?'on':''}" data-k="${k}" data-t="${id}">${lbl}</span>`).join("");
      h += `<div class="card ${r?'rated':''}" id="card_${k}">
        <div class="crow">
          <span class="slot">${c.slot}</span>
          <span class="reveal-id" id="rid_${k}"></span>
          <audio controls preload="none" src="${c.file}"></audio>
          <span style="flex:1"></span>
          <span class="stars">${stars}</span>
        </div>
        <div class="tags">${tags}</div>
        <input class="note" data-k="${k}" placeholder="note (optional)…" value="${(state.notes[k]||'').replace(/"/g,'&quot;')}">
      </div>`;
    });
    sec.innerHTML = h;
    root.appendChild(sec);
  });

  root.querySelectorAll(".star").forEach(el => el.onclick = () => {
    const k = el.dataset.k, v = +el.dataset.v;
    state.ratings[k] = (state.ratings[k] === v) ? 0 : v; save(); render();
  });
  root.querySelectorAll(".tag").forEach(el => el.onclick = () => {
    const k = el.dataset.k, id = el.dataset.t;
    const arr = state.tags[k] || (state.tags[k] = []);
    const i = arr.indexOf(id); i<0 ? arr.push(id) : arr.splice(i,1); save(); render();
  });
  root.querySelectorAll(".note").forEach(el => el.oninput = () => {
    state.notes[el.dataset.k] = el.value; save();
  });

  const total = TESTS.reduce((n,t)=>n+t.candidates.length,0);
  const done = Object.values(state.ratings).filter(v=>v>0).length;
  document.getElementById("prog").textContent = `${done} / ${total} rated`;
  if (state.revealed) doReveal(false);
}

function doReveal(toggle){
  if (toggle) { state.revealed = true; save(); }
  const agg = {}; IDS.forEach(id => agg[id] = {sum:0, n:0, tags:{}});
  TESTS.forEach((t, ti) => {
    const truth = JSON.parse(atob(t.truth_b64));
    t.candidates.forEach(c => {
      const id = truth[c.slot], k = key(ti, c.slot);
      const rid = document.getElementById("rid_"+k);
      if (rid){ rid.textContent = IDLABEL[id]; rid.style.display = "inline-block";
                rid.style.background = id==="base" ? "#3a2030" : "#203a2c"; }
      const r = state.ratings[k] || 0;
      if (r){ agg[id].sum += r; agg[id].n += 1; }
      (state.tags[k]||[]).forEach(tg => agg[id].tags[tg] = (agg[id].tags[tg]||0)+1);
    });
  });
  const rows = IDS.map(id => ({ id, avg: agg[id].n ? agg[id].sum/agg[id].n : 0, n: agg[id].n, tags: agg[id].tags }));
  const scored = rows.filter(r=>r.n>0).sort((a,b)=>b.avg-a.avg);
  const bestNonBase = scored.find(r=>r.id!=="base");
  const baseRow = rows.find(r=>r.id==="base");
  let html = `<div class="ttitle">📊 Results</div>`;
  if (bestNonBase){
    html += `<p>Recommended checkpoint: <span class="win">${IDLABEL[bestNonBase.id]}</span> `
          + `(avg ${bestNonBase.avg.toFixed(2)}/5).`;
    if (baseRow && baseRow.n && baseRow.avg >= bestNonBase.avg - 0.25)
      html += ` ⚠️ Heads-up: the <b>base</b> scored ${baseRow.avg.toFixed(2)} — nearly as high, so the LoRA may not be adding much.`;
    html += `</p>`;
  }
  html += `<table><tr><th>Candidate</th><th>Avg /5</th><th># rated</th><th>Flags</th></tr>`;
  rows.sort((a,b)=>b.avg-a.avg).forEach(r => {
    const tagstr = Object.entries(r.tags).map(([t,c])=>`<span class="pill">${t}×${c}</span>`).join(" ");
    html += `<tr><td>${IDLABEL[r.id]} ${r.id==="base"?'<span class="pill">control</span>':''}</td>`
          + `<td class="${(bestNonBase&&r.id===bestNonBase.id)?'win':''}">${r.n?r.avg.toFixed(2):'—'}</td>`
          + `<td>${r.n}</td><td>${tagstr||'—'}</td></tr>`;
  });
  html += `</table>`;
  const s = document.getElementById("summary");
  s.innerHTML = html; s.style.display = "block";
  if (toggle) s.scrollIntoView({behavior:"smooth"});
}

document.getElementById("revealBtn").onclick = () => {
  const total = TESTS.reduce((n,t)=>n+t.candidates.length,0);
  const done = Object.values(state.ratings).filter(v=>v>0).length;
  if (done < total && !confirm(`Only ${done}/${total} rated. Reveal anyway?`)) return;
  doReveal(true);
};
document.getElementById("resetBtn").onclick = () => {
  if (confirm("Clear all your ratings, tags and notes?")) { state = blank(); save(); render();
    document.getElementById("summary").style.display="none"; }
};
document.getElementById("exportBtn").onclick = () => {
  const out = { lskey: LSKEY, exported_at: new Date().toISOString(), revealed: state.revealed, tests: [] };
  TESTS.forEach((t, ti) => {
    const truth = state.revealed ? JSON.parse(atob(t.truth_b64)) : null;
    out.tests.push({ title: t.title, text: t.text, candidates: t.candidates.map(c => ({
      slot: c.slot, identity: truth ? truth[c.slot] : "hidden",
      rating: state.ratings[key(ti,c.slot)] || 0,
      tags: state.tags[key(ti,c.slot)] || [],
      note: state.notes[key(ti,c.slot)] || "" })) });
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "blind_results.json"; a.click();
};

render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
