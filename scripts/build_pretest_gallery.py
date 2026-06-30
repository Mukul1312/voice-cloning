"""
build_pretest_gallery.py — audition the pronunciation pre-test.
For each test sentence, lays out the spoken text, the tricky term(s) it stress-tests, and the seed-takes
side by side so you can hear how the clone handles each devotional/Sanskrit term and flag any that need
phonetic respelling. Output -> voice_lab/pretest/  (servable). Run after pulling out_pretest -> data/out/lecture1/pretest/.
  .venv/Scripts/python.exe scripts/build_pretest_gallery.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/out/lecture1/pretest"
OUT = ROOT / "voice_lab" / "pretest"
AUD = OUT / "audio"

# label -> (spoken text, terms being stress-tested)
TESTS = {
    "date":     ("May eighth, nineteen seventy five, Thursday.", "the spoken DATE (May 8th 1975 Thursday)"),
    "vow":      ("...except a little maha water with Tulasi.", "mahā-water, Tulasī"),
    "iskcon":   ("To open an ISKCON centre in Odisha.", "ISKCON, Odisha"),
    "prema":    ("...complete and uninterrupted Krishna prema.", "kṛṣṇa-prema"),
    "prasadam": ("He will not accept any Prasadam at night.", "Prasādam"),
    "lila":     ("But what an astonishing leela You are performing.", "līlā"),
    "sannyasi": ("...severe austerity in front of a sannyasi present here.", "sannyāsī"),
    "gopala":   ("Oh my Lord, Gopala! Please shower Your mercy.", "Gopāla"),
    "gauranga": ("...in the service of Sri Sri Guru and Gauranga. Hare Krishna.", "Śrī Śrī, Gaurānga, Hare Kṛṣṇa"),
}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    AUD.mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"(?P<label>\w+)__s(?P<seed>\d+)\.wav")
    by = {}
    for p in sorted(SRC.glob("*.wav")):
        m = pat.match(p.name)
        if not m:
            continue
        shutil.copyfile(p, AUD / p.name)
        by.setdefault(m["label"], []).append((int(m["seed"]), f"audio/{p.name}"))
    rows = []
    for label in TESTS:
        text, terms = TESTS[label]
        takes = sorted(by.get(label, []))
        rows.append(dict(label=label, text=text, terms=terms,
                         takes=[{"seed": s, "wav": w} for s, w in takes]))
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(rows, ensure_ascii=False))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT/'index.html'}  ({sum(len(r['takes']) for r in rows)} clips, {len(rows)} sentences)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronunciation pre-test</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e38;--ink:#e7e9ee;--mut:#9aa3b2;--acc:#7c9cff;--warn:#ffb84a;}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:22px 16px 80px}
  h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 16px;max-width:680px}
  .item{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:0 0 12px}
  .term{font-size:12px;color:var(--warn);font-weight:700;margin:0 0 3px}.txt{margin:0 0 9px;font-size:14px}
  .takes{display:flex;gap:8px;flex-wrap:wrap}.take{flex:1 1 150px}.take .s{font-size:11px;color:var(--mut)}
  audio{width:100%;height:32px}
</style></head><body><div class="wrap">
  <h1>🗣️ Pronunciation pre-test — his clone on the diary's terms</h1>
  <p class="sub">Each block stress-tests the tricky <b style="color:var(--warn)">term(s)</b> in a real diary sentence, across 5 seed-takes. Listen for the highlighted term: does he say it with dignity? Note any that come out wrong — we'll fix those with phonetic respelling before building the full Short.</p>
  <div id="list"></div>
</div>
<script>
const DATA = /*__DATA__*/;
document.getElementById("list").innerHTML = DATA.map(r=>`
  <div class="item">
    <div class="term">▶ testing: ${r.terms}</div>
    <div class="txt">"${r.text}"</div>
    <div class="takes">${r.takes.map(t=>`<div class="take"><div class="s">seed ${t.seed}</div><audio controls preload="none" src="${t.wav}"></audio></div>`).join("")}</div>
  </div>`).join("");
</script></body></html>
"""

if __name__ == "__main__":
    main()
