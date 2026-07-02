"""
fetch_idt.py — Stage 1 of the HINDI pipeline: scrape GGS Hindi lectures from ISKCON Desire Tree.

The taptajivanam fetcher (fetch_lecture.py) is a page-scraper that also pulls an on-page transcript.
ISKCON Desire Tree is a static PHP file-index (Andromeda 1.9.3.6) with NO transcripts — audio only —
so this fetcher just enumerates the directory listing and downloads the MP3s. Stdlib only: the index is
server-rendered HTML with stable, path-based, tokenless URLs, so no JS/headless/firecrawl is needed.

Recon (2026-07-01): 4 flat files, ~107 MB total (~2-4 h), Accept-Ranges: bytes (resumable), no auth,
robots.txt empty. Direct URL = the plain path, e.g.
  https://audio.iskcondesiretree.com/02_-_ISKCON_Swamis/.../Hindi_Lectures/GGM_Hindi_Lecture-01.mp3

Output (per lecture; mirrors data/lectures/<slug>/ one level down under the hi/ namespace — the additive
two-language layout, English left frozen):
  data/hi/lectures/<slug>/audio.mp3
  data/hi/lectures/<slug>/meta.json          # source, name, mp3_url, slug, bytes, lang
  (transcript.txt is produced LATER by the Hindi-ASR step — none exists in the archive)

RUN:
  python scripts/fetch_idt.py                 # default: the GGS Hindi_Lectures folder
  python scripts/fetch_idt.py --list          # print the files, don't download
  python scripts/fetch_idt.py --index "<other ?q=f&f=... folder URL>"   # e.g. later, the Oriya folder

NEXT: ffprobe each audio.mp3 for exact duration, then the Hindi-ASR transcription step.
"""
import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hi" / "lectures"
BASE = "https://audio.iskcondesiretree.com"
DEFAULT_INDEX = (BASE + "/index.php?q=f&f=%2F02_-_ISKCON_Swamis%2FISKCON_Swamis_-_D_to_P"
                 "%2FHis_Holiness_Gour_Govinda_Swami%2FHindi_Lectures")
# polite, identifiable UA (the site is donation-funded; be a good guest)
UA = {"User-Agent": "voice-cloning-seva-fetch/1.0 (personal devotional archival; info@dualite.dev)"}

MP3_HREF = re.compile(r'href="(/[^"]+?\.mp3)"', re.I)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def slugify(filename: str) -> str:
    stem = re.sub(r"\.mp3$", "", filename, flags=re.I)
    return re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower() or "lecture"


def list_mp3s(index_url: str):
    """Return [(display_name, download_url), ...] for every .mp3 linked on the index page."""
    html = fetch(index_url).decode("utf-8", errors="replace")
    hrefs = sorted(set(MP3_HREF.findall(html)))
    if not hrefs:
        sys.exit("No .mp3 links found — wrong folder URL, or the index layout changed.")
    out = []
    for href in hrefs:
        name = urllib.parse.unquote(href.rsplit("/", 1)[-1])
        # quote for safety (spaces etc.) without double-encoding already-% escaped chars
        url = urllib.parse.urljoin(BASE, urllib.parse.quote(href, safe="/%:._-~"))
        out.append((name, url))
    return out


def remote_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def download(url: str, dst: Path) -> int:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r, dst.open("wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    return dst.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEFAULT_INDEX, help="ISKCON DT folder index URL")
    ap.add_argument("--list", action="store_true", help="list files only, don't download")
    ap.add_argument("--delay", type=float, default=1.0, help="polite seconds between downloads")
    a = ap.parse_args()

    files = list_mp3s(a.index)
    print(f"found {len(files)} mp3(s):")
    for name, _ in files:
        print(f"  {name}")
    if a.list:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, (name, url) in enumerate(files):
        slug = slugify(name)
        d = OUT / slug
        d.mkdir(parents=True, exist_ok=True)
        dst = d / "audio.mp3"
        size = remote_size(url)
        if dst.exists() and size and dst.stat().st_size == size:
            print(f"[{i + 1}/{len(files)}] skip {slug} (complete, {size / 1e6:.1f} MB)")
        else:
            print(f"[{i + 1}/{len(files)}] downloading {slug} ...")
            got = download(url, dst)
            print(f"      {got / 1e6:.1f} MB")
        (d / "meta.json").write_text(json.dumps(
            {"source": "iskcondesiretree", "name": name, "mp3_url": url, "slug": slug,
             "bytes": dst.stat().st_size, "lang": "hi"}, indent=2), encoding="utf-8")
        total += dst.stat().st_size
        if i < len(files) - 1:
            time.sleep(a.delay)

    print(f"\nDONE -> data/hi/lectures/  ({len(files)} files, {total / 1e6:.1f} MB)")
    print("NEXT: ffprobe each audio.mp3 for exact duration, then Hindi-ASR transcription.")


if __name__ == "__main__":
    main()
