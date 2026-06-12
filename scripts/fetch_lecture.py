"""
fetch_lecture.py - Stage 1 of the taptajivanam.com data pipeline (stdlib only).

Given ONE lecture page URL, it:
  1. fetches the page HTML (urllib - no external tools, no PATH issues),
  2. pulls the direct MP3 url from the <audio><source> tag,
  3. extracts the official transcript from the page's justified <p> paragraphs,
  4. downloads the MP3 and saves the transcript.

Output (per lecture):
  data/lectures/<slug>/audio.mp3
  data/lectures/<slug>/transcript.txt    # raw official transcript (Eng + Sanskrit)
  data/lectures/<slug>/meta.json

RUN:
  python scripts/fetch_lecture.py "https://www.taptajivanam.com/view-archive.php?850000---I-and-Mine-and-Namabhasa-Stage&a_id=11"

NOTE: the transcript can contain brief questioner turns (not GGS). The diacritic
(Sanskrit) filter + speaker handling happen in Stage 2 (align_cut.py) + manual review.
"""
import argparse, html, json, re, sys, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "data" / "lectures"

# transcript paragraphs on this site are uniquely styled; matches ONLY the lecture body
P_RE      = re.compile(r'<p[^>]*dir="ltr"[^>]*text-align:\s*justify[^>]*>(.*?)</p>',
                       re.I | re.S)
SOURCE_RE = re.compile(r'<source[^>]+src="([^"]+\.mp3)"', re.I)
SPEAKER   = "Gour Govinda Swami:"


def slugify(url: str) -> str:
    q = urllib.parse.urlparse(url).query
    raw = q.split("&a_id")[0] if "&a_id" in q else q
    raw = re.sub(r"^\d+-+", "", urllib.parse.unquote(raw))     # strip leading "850000---"
    return re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower() or "lecture"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def strip_tags(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)              # <br> -> newline
    s = re.sub(r"<[^>]+>", "", s)                              # drop remaining tags
    s = html.unescape(s)
    s = s.replace(SPEAKER, "")                                # remove the speaker label text
    return re.sub(r"[ \t]+", " ", s).strip()


def extract_transcript(page: str) -> str:
    paras = [strip_tags(p) for p in P_RE.findall(page)]
    paras = [p for p in paras if p and p not in ("\xa0",)]    # drop empty / &nbsp; paragraphs
    if not paras:
        sys.exit("No transcript paragraphs found - page layout may have changed.")
    return "\n\n".join(paras).strip()


def extract_mp3(page: str, base_url: str) -> str:
    m = SOURCE_RE.search(page)
    if not m:
        sys.exit("No <source ...mp3> found - page layout may have changed.")
    src = html.unescape(m.group(1))                           # &amp; -> &
    return urllib.parse.urljoin(base_url, src)                # resolve relative -> absolute


def download(url: str, dst: Path) -> None:
    print(f"[3/3] Downloading MP3 -> {dst.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, dst.open("wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    print(f"      {dst.stat().st_size/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="taptajivanam lecture page URL")
    args = ap.parse_args()

    print("[1/3] Fetching page HTML...")
    page = fetch(args.url)

    print("[2/3] Extracting MP3 url + transcript...")
    mp3_url    = extract_mp3(page, args.url)
    transcript = extract_transcript(page)
    print(f"      mp3: {mp3_url}")
    print(f"      transcript: {len(transcript)} chars, ~{len(transcript.split())} words")

    slug = slugify(args.url)
    dst_dir = OUT / slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "transcript.txt").write_text(transcript, encoding="utf-8")
    (dst_dir / "meta.json").write_text(
        json.dumps({"url": args.url, "mp3_url": mp3_url, "slug": slug}, indent=2),
        encoding="utf-8")
    download(mp3_url, dst_dir / "audio.mp3")

    print(f"\nDONE -> data/lectures/{slug}/  (audio.mp3 + transcript.txt)")
    print("NEXT: skim transcript.txt (full lecture, English + Sanskrit), then Stage 2.")


if __name__ == "__main__":
    main()
