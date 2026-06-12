"""
classify_transliterate.py - Step 2a: validate D3 (verse filter) + D4 (IAST->ASCII)
on a real lecture transcript. Pure stdlib, no GPU, no installs.

For data/lectures/<slug>/transcript.txt it:
  1. splits into sentences,
  2. CLASSIFIES each as KEEP (English-with-terms) or DROP (chanted verse), using
     IAST-diacritic fraction + longest diacritic run + explicit [verse]/[Song]/citation
     markers  (D3),
  3. TRANSLITERATES kept sentences IAST->plain ASCII (Krishna, maya-moha)          (D4),
  4. writes a full report + the kept ASCII text, and prints samples to eyeball.

This is a VALIDATION tool: read the report, judge the calls, then we tune the
thresholds (or add a Sanskrit lexicon) before wiring up MFA alignment.

RUN:  python scripts/classify_transliterate.py i-and-mine-and-namabhasa-stage
      # tune: --frac 0.5 (verse if >= this fraction of words are Sanskrit)
      #       --run 4   (verse if this many diacritic-words appear in a row)
"""
import argparse, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LECT = ROOT / "data" / "lectures"

# D3 upgrade: English dictionary catches Sanskrit words that lack diacritics
WORDLIST_URL   = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
WORDLIST_CACHE = ROOT / "data" / "english_words.txt"
CONTRACTIONS   = {"ll", "ve", "re"}            # contraction tails to treat as English

# ---- D4: IAST -> ASCII map (pronounceable, Vaishnava-conventional) -----------
IAST_MAP = {
    "ā":"a","Ā":"A","ī":"i","Ī":"I","ū":"u","Ū":"U","ē":"e","Ē":"E","ō":"o","Ō":"O",
    "ṛ":"ri","Ṛ":"Ri","ṝ":"ri","Ṝ":"Ri","ḷ":"li","Ḷ":"Li","ḹ":"li","Ḹ":"Li",
    "ṅ":"n","Ṅ":"N","ñ":"n","Ñ":"N","ṇ":"n","Ṇ":"N","ṁ":"m","Ṁ":"M","ṃ":"m","Ṃ":"M",
    "ś":"sh","Ś":"Sh","ṣ":"sh","Ṣ":"Sh","ṭ":"t","Ṭ":"T","ḍ":"d","Ḍ":"D","ḥ":"h","Ḥ":"H",
    "ṟ":"r","ḻ":"l","ṉ":"n","’":"","‘":"","ʼ":"",
}
_TRANS = {ord(k): v for k, v in IAST_MAP.items()}
# D3: a word is "Sanskrit-ish" if it carries any of these diacritic letters
DIA = set(c for c in IAST_MAP if c not in ("’", "‘", "ʼ"))

MARKER_RE   = re.compile(r"\[(?:verse|song|[^\]]*?\d+\.\d+(?:\.\d+)?)", re.I)  # [verse]/[Song:..]/[..5.5.8]
BRACKET_RE  = re.compile(r"\[[^\]]*\]")     # editorial [...] - not spoken
WORD_RE     = re.compile(r"[^\s]+")


def transliterate(s: str) -> str:
    s = BRACKET_RE.sub("", s)               # drop [...] annotations (not spoken)
    return re.sub(r"\s{2,}", " ", s.translate(_TRANS)).strip()


def load_english() -> set:
    if not WORDLIST_CACHE.exists():
        WORDLIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading English wordlist (one-time, ~4 MB)...")
        req = urllib.request.Request(WORDLIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            WORDLIST_CACHE.write_bytes(r.read())
    return set(WORDLIST_CACHE.read_text(encoding="utf-8").split()) | CONTRACTIONS


def is_sanskrit(word: str, eng: set) -> bool:
    if any(ch in DIA for ch in word):                 # has IAST diacritic -> Sanskrit
        return True
    parts = [p for p in re.findall(r"[a-z]+", word.lower()) if len(p) >= 2]
    if not parts:                                      # numbers / single letters -> neutral
        return False
    return not all(p in eng for p in parts)            # any non-English alpha part -> Sanskrit


def sentences(text: str):
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def classify(sent: str, eng: set, frac_thr: float, run_thr: int):
    """Return (keep: bool, frac: float, max_run: int, reason: str)."""
    cited = bool(MARKER_RE.search(sent))              # has a scripture citation / [verse]
    spoken = BRACKET_RE.sub("", sent)                 # drop [...] citations for word stats
    words = WORD_RE.findall(spoken)
    if not words:
        return False, 0.0, 0, "empty"
    flags = [is_sanskrit(w, eng) for w in words]
    frac = sum(flags) / len(flags)
    run = mx = 0
    for f in flags:
        run = run + 1 if f else 0
        mx = max(mx, run)
    # a long Sanskrit run NEXT TO a scripture citation = a recited verse (vs a spoken term list)
    if mx >= run_thr and cited: return False, frac, mx, f"verse(run>={run_thr}+cited)"
    if frac >= frac_thr:        return False, frac, mx, f"frac>={frac_thr}"
    if mx   >= run_thr + 3:     return False, frac, mx, f"run>={run_thr+3}"
    return True, frac, mx, "english"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", default="i-and-mine-and-namabhasa-stage")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--run",  type=int,   default=4)
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")      # Sanskrit prints cleanly on Windows
    except Exception: pass

    src = LECT / args.slug / "transcript.txt"
    if not src.exists():
        sys.exit(f"Not found: {src}")
    sents = sentences(src.read_text(encoding="utf-8"))
    eng = load_english()
    print(f"English dictionary: {len(eng)} words")

    rows, keeps = [], []
    for s in sents:
        keep, frac, mx, reason = classify(s, eng, args.frac, args.run)
        ascii_txt = transliterate(s) if keep else ""
        rows.append((keep, frac, mx, reason, s, ascii_txt))
        if keep and ascii_txt:
            keeps.append(ascii_txt)

    # write artifacts
    out_dir = src.parent
    report = out_dir / "classify_report.txt"
    with report.open("w", encoding="utf-8") as f:
        for keep, frac, mx, reason, s, ascii_txt in rows:
            tag = "KEEP" if keep else "DROP"
            shown = ascii_txt if keep else s
            f.write(f"[{tag}] f={frac:.2f} r={mx} {reason:10} | {shown}\n")
    (out_dir / "clean_english.txt").write_text("\n".join(keeps), encoding="utf-8")

    # console summary + samples
    nk = sum(r[0] for r in rows)
    print(f"\n=== {args.slug}:  {len(rows)} sentences -> KEEP {nk} / DROP {len(rows)-nk} "
          f"(frac>={args.frac}, run>={args.run}) ===")
    print(f"report -> {report}\nclean  -> {out_dir/'clean_english.txt'}\n")

    def sample(pred, n, label):
        print(f"--- {label} (sample) ---")
        shown = 0
        for keep, frac, mx, reason, s, ascii_txt in rows:
            if pred(keep, frac, mx, reason) and shown < n:
                txt = (ascii_txt if keep else s)
                print(f"  f={frac:.2f} r={mx} {reason:9} | {txt[:140]}")
                shown += 1
        print()

    sample(lambda k,f,m,r: k, 8, "KEPT (transliterated ASCII)")
    sample(lambda k,f,m,r: not k, 8, "DROPPED (original)")
    sample(lambda k,f,m,r: 0.30 <= f <= 0.65, 8, "BORDERLINE (tune here)")
    print("NEXT: open classify_report.txt, judge the KEEP/DROP calls + the ASCII "
          "spellings, and tell me what to tune.")


if __name__ == "__main__":
    main()
