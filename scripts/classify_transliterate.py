"""
transliterate utility  (was Step-2a "classify + transliterate"; classify is RETIRED).

IAST -> plain readable ASCII for the VoxCPM text frontend:
  Kṛṣṇa -> Krishna,  māyā-moha -> maya-moha,  Śrīmad Bhāgavatam -> Shrimad Bhagavatam.

WHY a custom *lossy* map (and NOT indic-transliteration) — verified June 2026:
  indic-transliteration's schemes (HK / ITRANS / OPTITRANS / SLP1) are *reversible*
  romanizations. They preserve ṛ≠r, ṣ≠s by emitting capitals/digraphs (kRSNa, kRShNa,
  kfzRa) and even leave the anusvāra ṁ as a raw diacritic (puṁsaH). That is the OPPOSITE
  of what a TTS text frontend wants. We want a lossy collapse to clean, pronounceable,
  Vaishnava-conventional ASCII. Tested on the real transcript: this map leaves zero
  non-ASCII, while every indic-transliteration scheme produced uglier, still-non-ASCII text.

Sanskrit verse DETECTION / EXCISION was RETIRED (2026-06): GGS *speaks* (recites) his
verses, he does not melodically chant them (probe-verified via pitch contour), so we keep
them as his real voice. The old English-dictionary detector also mis-fired on loanwords
(krishna / bhakti ARE in English wordlists -> false negatives -> verse leaks) and is gone.

RUN (optional — dump the ASCII transcript to eyeball it):
  python scripts/classify_transliterate.py i-and-mine-and-namabhasa-stage
"""
import argparse, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LECT = ROOT / "data" / "lectures"

# IAST diacritics -> pronounceable ASCII (Vaishnava-conventional) + typographic cleanup.
IAST_MAP = {
    "ā":"a","Ā":"A","ī":"i","Ī":"I","ū":"u","Ū":"U","ē":"e","Ē":"E","ō":"o","Ō":"O",
    "ṛ":"ri","Ṛ":"Ri","ṝ":"ri","Ṝ":"Ri","ḷ":"li","Ḷ":"Li","ḹ":"li","Ḹ":"Li",
    "ṅ":"n","Ṅ":"N","ñ":"n","Ñ":"N","ṇ":"n","Ṇ":"N","ṁ":"m","Ṁ":"M","ṃ":"m","Ṃ":"M",
    "ś":"sh","Ś":"Sh","ṣ":"sh","Ṣ":"Sh","ṭ":"t","Ṭ":"T","ḍ":"d","Ḍ":"D","ḥ":"h","Ḥ":"H",
    "ṟ":"r","ḻ":"l","ṉ":"n","’":"","‘":"","ʼ":"",
    "–":"-","—":"-","“":'"',"”":'"',"…":"...",     # typographic punctuation -> ASCII
}
_TRANS = {ord(k): v for k, v in IAST_MAP.items()}

BRACKET_RE = re.compile(r"\[[^\]]*\]")             # editorial [...] annotations — not spoken


def transliterate(s: str) -> str:
    s = BRACKET_RE.sub("", s)                       # drop [...] annotations (not spoken)
    return re.sub(r"\s{2,}", " ", s.translate(_TRANS)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", default="i-and-mine-and-namabhasa-stage")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    src = LECT / args.slug / "transcript.txt"
    if not src.exists():
        sys.exit(f"Not found: {src}")
    ascii_txt = transliterate(src.read_text(encoding="utf-8"))
    out = src.parent / "transcript.ascii.txt"
    out.write_text(ascii_txt, encoding="utf-8")
    leftover = sorted(set(c for c in ascii_txt if ord(c) > 127))
    print(f"{src.name}: {len(ascii_txt)} chars -> {out.name}")
    print("non-ASCII remaining:", leftover or "none")
    print("\n" + ascii_txt[:400] + " ...")


if __name__ == "__main__":
    main()
