# Jupytext percent-format mirror of ggs_dataprep_colab.ipynb (for review/diff; run the .ipynb on Colab).

# %% [markdown]
# # GGS TTS Data-Prep Pipeline (Colab GPU) — VoxCPM2 LoRA dataset
# 
# Force-aligns the **accurate human transcript** of Gour Govinda Swami (GGS) English lectures to audio, keeps **only GGS** (drops questioner turns via diarization + speaker verification), cuts coherent sentence-boundary clips, and runs an **ASR round-trip CER/WER** QA gate. We **KEEP every word GGS says, including recited Sanskrit verses** (he speaks, not chants them) — there is **no verse excision anywhere**.
# 
# **Prototype on ONE lecture first** (fastbook Ch7 ethic), eyeball it, *then* run the **SCALE** section over all 10 lectures with a **whole-lecture GROUP split**.
# 
# ---
# ## HUMAN SETUP CHECKLIST (do these once, by hand, before running)
# 
# 1. **GPU runtime**: `Runtime > Change runtime type > GPU` (T4 is enough). The GPU-check cell asserts this.
# 2. **Hugging Face token + gated terms** (needed for pyannote diarization + the ECAPA embedding it pulls):
#    - Create a **READ** token at https://huggingface.co/settings/tokens
#    - While **logged in**, open the **community-1** model card https://huggingface.co/pyannote/speaker-diarization-community-1 and **accept the terms for EVERY gated repo the card lists as a requirement** (the card enumerates them; do not guess a specific segmentation repo — accept all that it links). If `from_pretrained` returns `None`, you missed one.
#    - Store the token as a **Colab secret** named `HF_TOKEN` (left sidebar key icon → toggle notebook access). The config cell reads it automatically and falls back to a masked prompt.
# 3. **Mount Drive + upload inputs**: put each lecture under
#    `MyDrive/ggs-voice-clone/lectures/<slug>/` containing `audio.mp3` and `transcript.txt`.
#    - **TRANSCRIPT CONTRACT (read this):** the questioner-drop strategy aligns the *whole* transcript and then keeps only words that fall inside GGS-only audio. That only works cleanly if `transcript.txt` is **GGS-only** OR **keeps speaker labels** so we can isolate GGS text before aligning. The old `fetch_lecture.py` *stripped* `Gour Govinda Swami:` / `Devotee:` labels — re-fetch **keeping the labels** (the finetune-plan TODO mandates this; a questioner clip already leaked once: #83). If labels are present, set `SPEAKER_LABEL_RE` in CONFIG and we split to GGS-only text before aligning; Stage-D intersection is then a second safety net. If your transcript is already GGS-only, leave `SPEAKER_LABEL_RE=None`. Either way Stage D prints how many words were dropped and **hard-warns if >15%** (that fraction signals questioner text the aligner smeared into GGS regions).
# 4. **Designate a clean GGS reference clip**: a few seconds of **GGS-only** speech (no questioner / music / applause), e.g. `MyDrive/ggs-voice-clone/refs/ggs_ref_01.wav`. Set its path in the **CONFIG** cell as `REF_GGS_CLIPS`. One clip is enough to start; add 2–3 spanning different lectures/rooms for a more robust centroid.
# 5. **Numbers in the transcript**: spell out digits (e.g. `1882` → `eighteen eighty two`) — uroman/MMS strips bare digits and that desyncs alignment. (The config cell prints any digit-bearing lines so you can fix them.)
# 
# **Run order:** install cell → **Runtime > Restart session** → GPU check → verify imports (+ Whisper smoke test) → mount Drive + CONFIG → prototype cells (Stages A–F) → inspect → SCALE section.

# %% Cell 1 — Install (SAFE ordering/pins, run FIRST then RESTART)
# =====================================================================================
# Cell 1 — DEPENDENCIES.  RUN THIS CELL, then Runtime > Restart session, then run below.
# -------------------------------------------------------------------------------------
# SAFE ORDERING (verified June 2026). The whole audio stack must stay on torch 2.8 so nothing
# triggers torchaudio>=2.9 (which removed AudioMetaData and crashes pyannote 4.0). The two
# things that historically clobbered that pin are: (a) ctc-forced-aligner floats torchcodec
# (its requirements pin NEITHER torch nor torchcodec -> pip would resolve torchcodec 0.8..0.14,
# which either bumps torch off 2.8 or re-introduces the std::bad_alloc segfault); (b) a stray
# `pip install -U torch`. We defeat (a) by installing the aligner with --no-deps (pulling only its
# pure-Python deps explicitly) AND by RE-PINNING the torch trio LAST so nothing can have moved it.
# =====================================================================================
!apt-get -qq update && apt-get -qq install -y ffmpeg   # audio decode for ctc-aligner + faster-whisper

# 1) numpy first, pinned to the range the torch-2.8 stack + numba/librosa agree on (keeps numba/llvmlite coherent):
!pip -q install "numpy>=2.1,<2.3"

# 2) Colab-safe torch trio (torch 2.8.0 + torchaudio 2.8.0 + torchcodec 0.7 = the non-segfaulting set).
#    Use the DEFAULT PyPI cu12x wheels (they match Colab's driver + bundled cuDNN-9); NO cu124 extra-index.
!pip -q install "torch==2.8.0" "torchaudio==2.8.0" "torchcodec==0.7"

# 3) Diarization + ECAPA speaker-verify (community-1 lives in pyannote.audio 4.x; speechbrain pinned explicitly):
!pip -q install "pyannote.audio>=4.0,<5.0" "speechbrain>=1.0.0"

# 4) PRIMARY forced aligner (MMS-300m). IMPORTANT: this step COMPILES a pybind11 C++ extension from
#    source (gcc is present on Colab GPU images; expect ~1-2 min). We install it with --no-deps so it
#    can NEVER touch torch/torchcodec, then add ONLY its pure-Python deps. Runtime is torchaudio-free
#    (it decodes via ffmpeg/torchcodec, not torchaudio's removed forced_align). Pin a known-good commit
#    so a future breaking commit can't change the build.
ALIGNER_SHA = "main"   # pin to a verified SHA (spec cites last-good 2026-04-15) if main ever breaks the build
!pip -q install --no-deps git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@{ALIGNER_SHA}
!pip -q install uroman nltk "transformers>=4.48" Unidecode

# 5) ASR round-trip QA: faster-whisper + a cuDNN-9 ctranslate2 build (>=4.5 matches torch-2.8's cuDNN-9):
!pip -q install "faster-whisper==1.2.0" "ctranslate2>=4.5" "jiwer==4.0.0"

# 6) Pure-Python audio I/O / QA metrics — installed with the numpy pin already fixed above:
!pip -q install "librosa>=0.11" soundfile

# 7) huggingface_hub: install withOUT -U so we keep the version pyannote.audio 4.0 resolved against
#    (a blanket `-U` can pull a hub newer than pyannote expects and break Pipeline.from_pretrained token handling):
!pip -q install huggingface_hub

# 8) FINAL RE-PIN of the torch trio: assert nothing above moved it off the non-segfaulting set.
#    (This is the belt-and-suspenders that makes step 4's --no-deps bulletproof.)
!pip -q install "torch==2.8.0" "torchaudio==2.8.0" "torchcodec==0.7"

print("\n>>> Cell 1 done. Now: Runtime > Restart session, then run every cell BELOW this one.")
print(">>> Do NOT re-run this install cell after the restart, and do NOT `pip install -U torch/torchcodec` "
      "(re-introduces 2.9 / torchcodec>=0.8 -> pyannote crash or segfault).")

# %% Cell 2 — GPU + runtime sanity check (run AFTER restart)
# =====================================================================================
# Cell 2 — GPU / RUNTIME CHECK. Colab can silently hand you a CPU runtime; fail fast.
# Also expose torch's bundled cuDNN-9 libs HERE (early, before any cell loads ctranslate2/
# faster-whisper) so the dynamic loader sees them at WhisperModel construction time.
# =====================================================================================
import subprocess, os, glob
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
      or "!! nvidia-smi found no GPU — set Runtime > Change runtime type > GPU and reconnect.")

import torch
print(f"torch            : {torch.__version__}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device           : {torch.cuda.get_device_name(0)}")
    print(f"torch CUDA build : {torch.version.cuda}")
    print(f"cuDNN            : {torch.backends.cudnn.version()}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM             : {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")

assert torch.cuda.is_available(), "No CUDA GPU attached — this pipeline needs a GPU runtime."
if not torch.__version__.startswith("2.8."):
    print(f"\n!! WARNING: torch is {torch.__version__}, expected 2.8.x. The install cell was skipped "
          "or torch got clobbered (a stray `pip install -U torch` re-introduces 2.9 -> pyannote crashes "
          "with `torchaudio has no attribute AudioMetaData`). Re-run Cell 1 and Restart session.")
else:
    print("\nOK: torch 2.8.x — matches the pyannote-4.0 pinned stack.")

# cuDNN-9: torch 2.8 ships the .so's; expose them NOW (before faster_whisper/ctranslate2 import anywhere)
# so ctranslate2 doesn't hit 'Unable to load libcudnn_ops.so.9'. Must be set in the process BEFORE the
# first WhisperModel(...) — Cell 3's smoke test verifies it actually worked.
cudnn_dirs = sorted({os.path.dirname(p) for p in glob.glob(
    os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cudnn", "lib", "*.so*"))})
if cudnn_dirs:
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(cudnn_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    print("LD_LIBRARY_PATH += torch cuDNN libs (fixes ctranslate2 cuDNN-9 load).")

# %% Cell 3 — Verify conflict-prone imports + Whisper cuDNN smoke test (run AFTER restart)
# =====================================================================================
# Cell 3 — POST-INSTALL VERIFICATION. Proves the tricky stack imports on this torch, that the
# torch/torchcodec pins held, and that ctranslate2/faster-whisper can actually LOAD a CUDA model
# (cuDNN-9). We surface a cuDNN load failure HERE (seconds) instead of 30 min into the scale loop.
# =====================================================================================
import importlib, torch, torchaudio
print(f"torch {torch.__version__} / torchaudio {torchaudio.__version__}")
assert torchaudio.__version__.startswith("2.8."), \
    f"torchaudio {torchaudio.__version__} != 2.8.x -> pyannote AudioMetaData crash risk."

import torchcodec
assert torchcodec.__version__.startswith("0.7"), \
    f"torchcodec {torchcodec.__version__} != 0.7.x -> the aligner/pyannote install floated it; re-run Cell 1 step 8."

import numpy, transformers, huggingface_hub
from pyannote.audio import Pipeline                              # diarization
from speechbrain.inference.speaker import EncoderClassifier      # ECAPA speaker-verify (1.0+ path)
import ctc_forced_aligner                                        # PRIMARY forced aligner (C++ ext built in Cell 1)
from faster_whisper import WhisperModel                          # ASR round-trip
import jiwer, librosa                                            # CER/WER + QA metrics
print("imports OK:",
      "| pyannote.audio", importlib.import_module('pyannote.audio').__version__,
      "| speechbrain", importlib.import_module('speechbrain').__version__,
      "| transformers", transformers.__version__,
      "| hub", huggingface_hub.__version__,
      "| numpy", numpy.__version__,
      "| librosa", librosa.__version__)

# cuDNN-9 SMOKE TEST: build a tiny CUDA Whisper so a libcudnn load failure surfaces now, not later.
# (LD_LIBRARY_PATH was already set in Cell 2 before this import.)
try:
    _m = WhisperModel("tiny", device="cuda", compute_type="int8")
    del _m
    print("faster-whisper/ctranslate2 CUDA load OK (cuDNN-9 resolved).")
except Exception as e:
    print("!! WhisperModel CUDA smoke test FAILED:", repr(e))
    print("   If this is 'Unable to load libcudnn_ops.so.9', the LD_LIBRARY_PATH from Cell 2 did not take "
          "effect in this process. Restart the session and run Cell 2 then Cell 3 again (do not skip Cell 2), "
          "or fall back to compute_type='int8_float16'.")

# %% Cell 4 — Mount Drive + CONFIG (the ONE place you edit)
# =====================================================================================
# Cell 4 — Mount Google Drive + CONFIG.  EDIT THE MARKED VALUES; everything else reads from here.
# Drive persists inputs AND outputs across runtime resets. We CUT clips to fast local /content
# scratch and batch-sync to Drive at the end (thousands of tiny FUSE writes are slow/flaky).
# =====================================================================================
from google.colab import drive
from pathlib import Path
import re

drive.mount("/content/drive")

# ----------------------------- EDIT THESE -----------------------------
DRIVE_PROJ   = Path("/content/drive/MyDrive/ggs-voice-clone")   # your project root on Drive
SLUG         = "i-and-mine-and-namabhasa-stage"                  # the ONE lecture to prototype on
REF_GGS_CLIPS = [                                               # clean GGS-ONLY reference clip(s)
    str(DRIVE_PROJ / "refs" / "ggs_ref_01.wav"),
    # str(DRIVE_PROJ / "refs" / "ggs_ref_02.wav"),                # add more spanning lectures/rooms
]
LANGUAGE     = "eng"            # ISO-639-3 for the aligner (English base; verses spoken in-stream)
# TRANSCRIPT speaker labels: if transcript.txt KEEPS labels, set a regex that matches a line/turn
# header for a NON-GGS speaker so we can drop questioner TEXT before aligning. Leave None if the
# transcript is already GGS-only. Example for 'Devotee:' / 'Question:' headers:
SPEAKER_LABEL_RE = None        # e.g. re.compile(r'^\s*(devotee|question|questioner|audience)\s*:', re.I)
GGS_LABEL_RE     = None        # e.g. re.compile(r'^\s*(gour govinda swami|ggs|maharaja)\s*:', re.I)
MAX_WORD_DROP_FRAC = 0.15      # Stage D hard-warns if more than this fraction of words drop (questioner smear)
# diarization speaker-count bounds (solo lecture + occasional questioners):
MIN_SPEAKERS, MAX_SPEAKERS = 1, 4
# speaker-verify: raw-cosine threshold (CALIBRATE on your data; 0.40 raw ~= 0.70 rescaled):
COSINE_THR   = 0.40
COSINE_MARGIN = 0.10           # min gap between best and 2nd-best cluster cosine (else ambiguous -> flag)
MIN_TURN_SEC = 1.0             # turns shorter than this are too short to embed reliably
# clip grouping (matches the local align_cut.py recipe):
MIN_SEC, MAX_SEC = 9.0, 20.0
# ASR QA thresholds (STARTING POINTS — re-set from the printed distributions):
MAX_CER_CORE, MAX_WER = 0.30, 0.60
MAX_LEAD_SIL, MAX_TRAIL_SIL = 0.6, 1.0
ASR_MODEL    = "large-v3"
# ----------------------------------------------------------------------

# Derived paths (do not edit). Lecture wavs go under a _work/ subdir so they are NOT synced to Drive.
DRIVE_LECTURES = DRIVE_PROJ / "lectures"           # inputs: <slug>/audio.mp3 + transcript.txt
DRIVE_OUT      = DRIVE_PROJ / "out";   DRIVE_OUT.mkdir(parents=True, exist_ok=True)
LOCAL_OUT      = Path("/content/out"); LOCAL_OUT.mkdir(parents=True, exist_ok=True)
LOCAL_WORK     = Path("/content/_work"); LOCAL_WORK.mkdir(parents=True, exist_ok=True)  # full-lecture wavs (not synced)

assert DRIVE_LECTURES.exists(), f"Put lectures under {DRIVE_LECTURES} (each <slug>/audio.mp3 + transcript.txt)."
print("lectures found:", sorted(p.name for p in DRIVE_LECTURES.iterdir() if p.is_dir()))
assert (DRIVE_LECTURES / SLUG / "transcript.txt").exists(), f"Missing transcript for SLUG={SLUG}"

# HF token (Colab secret 'HF_TOKEN' preferred; else masked prompt).
def get_hf_token() -> str:
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok: return tok
    except Exception:
        pass
    import getpass
    return getpass.getpass("Paste your Hugging Face READ token: ").strip()
HF_TOKEN = get_hf_token()

# Sanity: flag bare digits in the transcript (uroman strips them -> alignment desync).
_raw = (DRIVE_LECTURES / SLUG / "transcript.txt").read_text(encoding="utf-8")
_digit_lines = [ln for ln in _raw.splitlines() if re.search(r"\d", ln)]
if _digit_lines:
    print(f"\n!! {len(_digit_lines)} transcript line(s) contain digits — SPELL THEM OUT before aligning:")
    for ln in _digit_lines[:8]:
        print("   ", ln.strip()[:100])
else:
    print("\nNo bare digits in transcript — good for uroman/MMS alignment.")

# Sanity: report whether speaker labels are present so you know if SPEAKER_LABEL_RE should be set.
if SPEAKER_LABEL_RE is None and GGS_LABEL_RE is None:
    _hdrish = [ln for ln in _raw.splitlines() if re.match(r"^\s*[A-Z][A-Za-z .]{1,30}:", ln)]
    if _hdrish:
        print(f"\n!! {len(_hdrish)} line(s) look like 'Speaker:' headers but SPEAKER_LABEL_RE/GGS_LABEL_RE are None.")
        print("   If this transcript still contains questioner turns, set those regexes (see checklist item 3),")
        print("   else questioner TEXT will be aligned and only dropped by the Stage-D geometry safety net.")
        for ln in _hdrish[:6]:
            print("   ", ln.strip()[:100])
    else:
        print("\nNo 'Speaker:' headers detected — transcript treated as GGS-only (Stage D is the safety net).")
print("\nCONFIG OK.")

# %% [markdown]
# ## Pipeline order — and the timestamp intersection
# 
# We must keep **only GGS** and **preserve the accurate transcript**. That dictates this order:
# 
# 1. **Stage A — Diarize** the full lecture → speaker turns `{start, end, speaker}`. Labels are arbitrary (`SPEAKER_00`…); this only finds *who-speaks-when*, not *who is GGS*.
# 2. **Stage B — Speaker-verify** which diarization cluster is GGS, by cosine-matching each cluster against your clean GGS reference embedding (ECAPA). Output: the set of **GGS-only time intervals**. Short GGS turns are skipped only for the *cluster-ID vote* but are still included when building the GGS intervals.
# 3. **Stage C — Forced-align** the **GGS transcript** to the audio with MMS-300m CTC. If the transcript keeps speaker labels we first restrict it to GGS-only text (`SPEAKER_LABEL_RE`/`GGS_LABEL_RE`); otherwise we align the whole (GGS-only) transcript. We never re-transcribe — we get **per-word timestamps** for the known text.
# 4. **Stage D — Intersect** (the key step): keep a word **only if its `[start,end]` span lies (near-fully) inside a GGS-only interval**. This is how diarization+verification actually *drops questioner words* — by geometry on the timeline, not by editing text. Verses stay (they're GGS, inside GGS intervals). A hard sanity gate warns if too many words drop (signals questioner text smeared into GGS regions).
# 5. **Stage E — Group** the surviving GGS words into **sentence-boundary clips** (~9–20 s coherent ideas), pad lead-in, cut 16 kHz mono WAVs with **both-end silence trim**, write `train.jsonl` (`{audio, text, duration}`) where `duration` is read from the **actual cut wav**. A clip is only emitted if **all its words are GGS-only**, and the trailing pad is **clamped to the GGS-interval boundary** so no clip bleeds into the following questioner turn.
# 6. **Stage F — ASR round-trip QA**: re-transcribe each clip with faster-whisper, score CER/WER vs the clip's known text (Sanskrit-masked **cer_core** for pass/fail, raw CER reported as `sanskrit_load`). Output `train.clean.jsonl` + `qa_asr_report.tsv`.
# 
# Diarize/verify **before** slicing is essential: we need the GGS-only intervals *before* deciding which aligned words to cut. Forced-align runs on the GGS transcript independently of diarization; the **intersection** in Stage D is where the two timelines meet.

# %% Cell 5 — transliterate() (pasted verbatim) + shared helpers
# =====================================================================================
# Cell 5 — Repo conventions reused VERBATIM: transliterate() (IAST -> lossy readable ASCII)
# and the bracket-stripper. KEEPS verses; only collapses diacritics + drops [editorial] notes.
# (Copied from scripts/classify_transliterate.py — do NOT re-litigate the map.)
# Plus shared clip/cut helpers (verbatim from align_cut.py) and a GGS-only transcript splitter.
# =====================================================================================
import re, json, subprocess
from pathlib import Path
import librosa  # used for real-duration readback

IAST_MAP = {
    "ā":"a","Ā":"A","ī":"i","Ī":"I","ū":"u","Ū":"U",
    "ē":"e","Ē":"E","ō":"o","Ō":"O",
    "ṛ":"ri","Ṛ":"Ri","ṝ":"ri","Ṝ":"Ri","ḷ":"li","Ḷ":"Li",
    "ḹ":"li","Ḹ":"Li",
    "ṅ":"n","Ṅ":"N","ñ":"n","Ñ":"N","ṇ":"n","Ṇ":"N",
    "ṁ":"m","Ṁ":"M","ṃ":"m","Ṃ":"M",
    "ś":"sh","Ś":"Sh","ṣ":"sh","Ṣ":"Sh","ṭ":"t","Ṭ":"T",
    "ḍ":"d","Ḍ":"D","ḥ":"h","Ḥ":"H",
    "ṟ":"r","ḻ":"l","ṉ":"n",
    "’":"","‘":"","ʼ":"",
    "–":"-","—":"-","“":'"',"”":'"',"…":"...",
}
_TRANS = {ord(k): v for k, v in IAST_MAP.items()}
BRACKET_RE = re.compile(r"\[[^\]]*\]")     # editorial [...] annotations — not spoken

def transliterate(s: str) -> str:
    s = BRACKET_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s.translate(_TRANS)).strip()

def ggs_only_text(raw: str, speaker_re, ggs_re) -> str:
    """If the transcript keeps speaker labels, keep ONLY GGS turns' text (drop questioner TEXT before
    aligning). Turns are delimited by lines starting with a 'Speaker:' header. If neither regex is set,
    return the raw text unchanged (transcript is treated as GGS-only). This is the FIRST line of defense;
    Stage D's geometry is the second."""
    if speaker_re is None and ggs_re is None:
        return raw
    header_re = re.compile(r"^\s*([A-Za-z][A-Za-z .]{1,30}):\s*(.*)$")
    keep, cur_is_ggs = [], True   # default to GGS until a header says otherwise
    for ln in raw.splitlines():
        m = header_re.match(ln)
        if m:
            head = m.group(0)
            is_non_ggs = bool(speaker_re and speaker_re.match(ln))
            is_ggs     = bool(ggs_re and ggs_re.match(ln))
            if is_ggs:
                cur_is_ggs = True
            elif is_non_ggs:
                cur_is_ggs = False
            else:
                # unknown speaker header: treat as non-GGS to be safe (drop it)
                cur_is_ggs = False
            if cur_is_ggs:
                keep.append(m.group(2))   # text after 'GGS:' on the same line
        elif cur_is_ggs:
            keep.append(ln)
    return "\n".join(keep)

# --- clip recipe constants (verbatim from align_cut.py) ---
HARD_MIN, HARD_MAX = 4.0, 30.0
LEAD_PAD = 0.15      # secs lead-in so the first word's onset isn't clipped
TAIL_PAD = 0.15      # secs after the last word (before trailing-silence trim)

def to_wav_16k(src: Path, dst: Path) -> Path:
    """mp3/any -> 16k mono wav (matches align_cut.to_wav)."""
    if not dst.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst

def cut(src: Path, start: float, end: float, dst: Path):
    """Cut [start,end] and trim silence from BOTH ends to <0.1s (verbatim ffmpeg recipe)."""
    sr = "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB"
    trim = f"{sr},areverse,{sr},areverse"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-af", trim, "-ac", "1", "-ar", "16000", str(dst)], check=True)

def wav_duration(path: Path) -> float:
    """REAL duration of the written wav (after silence trim). Used for the manifest 'duration' field
    so it matches what VoxCPM will actually see (the requested end-start over-counts the trimmed silence)."""
    return float(librosa.get_duration(path=str(path)))

def build_clips(words, min_sec, max_sec):
    """Group words into clips that END AT A SENTENCE BOUNDARY (.?!), each ~min..max sec
    (a coherent 2-3 sentence idea). Returns [(start_idx, end_idx)] into `words`. (verbatim)"""
    clips, cur = [], []
    def flush():
        if cur:
            s, e = cur[0], cur[-1]
            if HARD_MIN <= words[e]["end"] - words[s]["start"] <= HARD_MAX:
                clips.append((s, e))
        cur.clear()
    for idx, w in enumerate(words):
        cur.append(idx)
        dur = words[cur[-1]]["end"] - words[cur[0]]["start"]
        ends_sentence = w["text"].rstrip().endswith((".", "?", "!"))
        if dur >= max_sec or (ends_sentence and dur >= min_sec):
            flush()
    flush()
    return clips

print("transliterate() + clip recipe + GGS-only splitter loaded. Quick check:")
print("  ", transliterate("śrīmad Bhāgavatam: Kṛṣṇa, māyā-moha [aside]"))

# %% Cell 6 — Stage A: diarize the lecture
# =====================================================================================
# Cell 6 — STAGE A: DIARIZATION (pyannote.audio 4.0, community-1) on GPU.
# Finds speaker turns. Labels are arbitrary; Stage B decides which cluster is GGS.
# We load audio in-memory (torchaudio.load -> dict) to BYPASS torchcodec (segfault/ffmpeg-load issues).
# =====================================================================================
import torch, torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pipeline(hf_token: str) -> Pipeline:
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token)   # 4.0: token= (not use_auth_token=)
    if pipe is None:
        raise RuntimeError(
            "Pipeline.from_pretrained returned None. Open the community-1 model card while LOGGED IN "
            "(https://huggingface.co/pyannote/speaker-diarization-community-1) and ACCEPT THE TERMS FOR "
            "EVERY GATED REPO IT LISTS AS A REQUIREMENT — the card enumerates them. Then use a valid READ token.")
    pipe.to(torch.device(DEVICE))
    print(f"[diarize] pipeline ready on {DEVICE}")
    return pipe

def load_audio_in_memory(wav_path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(wav_path)                 # (channels, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)             # downmix mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr); sr = target_sr
    return {"waveform": wav, "sample_rate": sr}

def diarize(pipe, wav_path, min_speakers=1, max_speakers=4, exclusive=True):
    """Return (turns, speaker_seconds). exclusive=True -> non-overlapping turns (STT-friendly)."""
    audio = load_audio_in_memory(wav_path)
    with ProgressHook() as hook:
        out = pipe(audio, hook=hook, min_speakers=min_speakers, max_speakers=max_speakers)
    ann = out.exclusive_speaker_diarization if exclusive else out.speaker_diarization
    turns, secs = [], {}
    for seg, _track, spk in ann.itertracks(yield_label=True):
        turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": spk})
        secs[spk] = secs.get(spk, 0.0) + float(seg.duration)
    turns.sort(key=lambda t: t["start"])
    return turns, secs

# --- run on the prototype lecture ---
lec_dir = DRIVE_LECTURES / SLUG
wav16   = to_wav_16k(lec_dir / "audio.mp3", LOCAL_WORK / f"{SLUG}.wav")   # 16k mono, local scratch (_work, not synced)

diar_pipe = load_pipeline(HF_TOKEN)
turns, speaker_seconds = diarize(diar_pipe, str(wav16),
                                 min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS, exclusive=True)
print(f"\n[diarize] {len(turns)} turns; per-speaker talk time:")
for spk, s in sorted(speaker_seconds.items(), key=lambda kv: -kv[1]):
    print(f"   {spk}: {s:8.1f}s ({s/60:5.1f} min)")
print("[diarize] GGS is almost certainly the speaker with the MOST talk time (Stage B confirms).")

# %% Cell 7 — Stage B: verify which cluster is GGS → GGS-only intervals
# =====================================================================================
# Cell 7 — STAGE B: SPEAKER VERIFICATION (ECAPA-TDNN). Decide which diarization cluster is GGS
# by cosine-matching each cluster's audio to your clean GGS reference embedding.
# Output: GGS_INTERVALS = merged [start,end] spans where GGS speaks (used by the Stage-D intersect).
# We ONLY bridge a small gap between two GGS turns if NO other speaker's turn falls in that gap
# (otherwise a questioner blurt in a <=MERGE_GAP gap would be swallowed into a 'GGS' interval).
# =====================================================================================
import torch, torchaudio
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from collections import defaultdict

TARGET_SR = 16000
MERGE_GAP = 0.25
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/content/pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE})

def load_wav_16k_mono(path, start=None, end=None):
    if start is not None and end is not None:
        sr = torchaudio.info(path).sample_rate
        wav, sr = torchaudio.load(path, frame_offset=int(start*sr), num_frames=int((end-start)*sr))
    else:
        wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    return wav

@torch.no_grad()
def embed_wav(wav_1xT):
    emb = classifier.encode_batch(wav_1xT.to(DEVICE)).squeeze(0).squeeze(0)   # (192,)
    return F.normalize(emb, dim=0)

def cosine(a, b):
    return float(torch.dot(a, b))

def build_ggs_intervals(turns, ggs_speaker, merge_gap=MERGE_GAP):
    """Merged [start,end] spans of the GGS cluster. Bridge a <=merge_gap gap ONLY when no other
    speaker's turn intersects that gap (so a questioner in the gap is NOT absorbed as GGS)."""
    ggs = sorted((t["start"], t["end"]) for t in turns if t["speaker"] == ggs_speaker)
    other = sorted((t["start"], t["end"]) for t in turns if t["speaker"] != ggs_speaker)
    def gap_has_other(g0, g1):
        return any(o0 < g1 and o1 > g0 for o0, o1 in other)   # any non-GGS turn overlapping the gap
    intervals = []
    for s, e in ggs:
        if intervals and (s - intervals[-1][1]) <= merge_gap and not gap_has_other(intervals[-1][1], s):
            intervals[-1] = (intervals[-1][0], e)
        else:
            intervals.append((s, e))
    return intervals

# 1) Build the GGS reference centroid from your clean clip(s).
for p in REF_GGS_CLIPS:
    assert Path(p).exists(), f"Reference clip not found: {p} (set REF_GGS_CLIPS in CONFIG)."
ggs_ref = F.normalize(torch.stack([embed_wav(load_wav_16k_mono(p)) for p in REF_GGS_CLIPS]).mean(0), dim=0)

# 2) Score each DIARIZATION CLUSTER (not each turn) against the reference: mean cosine over its turns.
#    Embedding whole clusters is robust; we skip sub-MIN_TURN_SEC turns (too short to embed reliably) —
#    this skip affects ONLY the cluster-ID vote, NOT the interval construction below.
sums, counts = defaultdict(float), defaultdict(int)
for t in turns:
    if (t["end"] - t["start"]) < MIN_TURN_SEC:
        continue
    sim = cosine(embed_wav(load_wav_16k_mono(str(wav16), t["start"], t["end"])), ggs_ref)
    sums[t["speaker"]] += sim; counts[t["speaker"]] += 1
speaker_cos = {spk: sums[spk]/counts[spk] for spk in counts}
print("[verify] per-cluster mean cosine to GGS reference:")
for spk, c in sorted(speaker_cos.items(), key=lambda kv: -kv[1]):
    print(f"   {spk}: cos={c:+.3f}   talk={speaker_seconds.get(spk,0)/60:5.1f} min")

ggs_speaker = max(speaker_cos, key=speaker_cos.get) if speaker_cos else None
assert ggs_speaker is not None, "No cluster long enough to verify — check the reference clip / diarization."
ranked = sorted(speaker_cos.values(), reverse=True)
margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
print(f"\n[verify] GGS cluster = {ggs_speaker} (cos={speaker_cos[ggs_speaker]:+.3f}, margin={margin:+.3f}). "
      "Expect it to be BOTH the highest-cosine AND the longest-talking cluster.")
if speaker_cos[ggs_speaker] < COSINE_THR:
    print(f"!! WARNING: top cosine {speaker_cos[ggs_speaker]:.3f} < COSINE_THR {COSINE_THR}. "
          "Reference clip may be contaminated/short — eyeball before trusting.")
if len(ranked) > 1 and margin < COSINE_MARGIN:
    print(f"!! WARNING: margin to 2nd-best cluster {margin:.3f} < COSINE_MARGIN {COSINE_MARGIN} — ambiguous; eyeball.")
top_talk = max(speaker_seconds, key=speaker_seconds.get)
if top_talk != ggs_speaker:
    print(f"!! WARNING: argmax-cosine ({ggs_speaker}) != max-talk-time ({top_talk}); inspect before trusting.")

# 3) GGS-only intervals = merged turns of the GGS cluster (gap-bridged ONLY when no other speaker is in the gap).
GGS_INTERVALS = build_ggs_intervals(turns, ggs_speaker)
ggs_total = sum(e - s for s, e in GGS_INTERVALS)
print(f"[verify] {len(GGS_INTERVALS)} merged GGS-only intervals, {ggs_total/60:.1f} min total kept.")

# %% Cell 8 — Stage C: forced-align the KNOWN GGS transcript (never re-transcribe)
# =====================================================================================
# Cell 8 — STAGE C: FORCED ALIGNMENT (ctc-forced-aligner, MMS-300m). Aligns YOUR accurate
# transcript to audio -> per-word timestamps. NEVER re-transcribes; verses preserved.
# If the transcript keeps speaker labels we FIRST restrict to GGS-only text (ggs_only_text),
# so questioner TEXT is not forced onto questioner AUDIO (which would drag GGS word boundaries).
# =====================================================================================
import soundfile as sf, numpy as np
from ctc_forced_aligner import (
    load_audio, load_alignment_model, generate_emissions,
    preprocess_text, get_alignments, get_spans, postprocess_results)

ALN_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
alignment_model, alignment_tokenizer = load_alignment_model(DEVICE, dtype=ALN_DTYPE)

def align_known_transcript(audio_path, transcript, language="eng", batch_size=8):
    """Force-align KNOWN transcript -> [{word,start,end,score}] (seconds). Does NOT alter text.
    audio_path is the 16k mono wav from to_wav_16k (already 16k mono, so no re-resample needed)."""
    audio_waveform = load_audio(audio_path, alignment_model.dtype, alignment_model.device)
    emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=batch_size)
    tokens_starred, text_starred = preprocess_text(
        transcript, romanize=True, language=language, split_size="word")  # romanize REQUIRED for MMS
    segments, scores, blank = get_alignments(emissions, tokens_starred, alignment_tokenizer)
    spans = get_spans(tokens_starred, segments, blank)
    raw = postprocess_results(text_starred, spans, stride, scores)        # start/end already in SECONDS
    words = []
    for w in raw:
        tok = w["text"]
        if tok in ("<star>", ""):                                         # wildcard/padding, skip
            continue
        words.append({"word": tok, "start": float(w["start"]), "end": float(w["end"]),
                      "score": float(w.get("score", 0.0))})
    return words

# Restrict to GGS-only TEXT (if labels present), then transliterate (IAST -> ASCII), then align.
_raw_txt = (lec_dir / "transcript.txt").read_text(encoding="utf-8")
_ggs_txt = ggs_only_text(_raw_txt, SPEAKER_LABEL_RE, GGS_LABEL_RE)
if SPEAKER_LABEL_RE is not None or GGS_LABEL_RE is not None:
    print(f"[align] speaker labels present -> kept {len(_ggs_txt.split())}/{len(_raw_txt.split())} "
          f"words as GGS-only TEXT before aligning.")
transcript_ascii = transliterate(_ggs_txt.replace("\n", " "))
words_aln = align_known_transcript(str(wav16), transcript_ascii, language=LANGUAGE, batch_size=8)
# normalize key 'word' -> 'text' so build_clips (which reads w['text']) works unchanged.
words = [{"text": w["word"], "start": w["start"], "end": w["end"], "score": w["score"]} for w in words_aln]

print(f"[align] {len(words)} words aligned. First 8:")
for w in words[:8]:
    print(f"   {w['start']:7.3f}-{w['end']:7.3f}  {w['score']:.2f}  {w['text']}")
dur_total = librosa.get_duration(path=str(wav16))
assert words and words[0]["start"] >= -0.05 and words[-1]["end"] <= dur_total + 0.1, "alignment out of bounds"
long_w = [w for w in words if (w["end"] - w["start"]) > 3.0]
print(f"[align] words >3s (inspect — likely pauses/recited-verse lines, KEEP them): {len(long_w)}")

# %% Cell 9 — Stage D: intersect aligned words with GGS-only intervals
# =====================================================================================
# Cell 9 — STAGE D: THE INTERSECTION. Keep a word ONLY if (near-)ALL of its [start,end] lies INSIDE a
# GGS-only interval. This is how questioner turns get dropped — by timeline geometry, not by editing
# text. We require OVERLAP_FRAC>=0.9 (not 0.5): we are CUTTING clips, so a boundary word half-inside a
# questioner turn is cheap to lose and expensive to keep wrong (forced-align and diarization boundaries
# come from different models and are not co-calibrated). Hard sanity gate on the drop fraction.
# =====================================================================================
OVERLAP_FRAC = 0.9   # >=90% of the word's duration must fall inside a GGS interval to keep it

def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def word_is_ggs(w, intervals):
    wd = max(1e-6, w["end"] - w["start"])
    # intervals are sorted & disjoint; sum overlap (a word may straddle a merge gap).
    ov = sum(_overlap(w["start"], w["end"], s, e) for s, e in intervals)
    return (ov / wd) >= OVERLAP_FRAC

for w in words:
    w["is_ggs"] = word_is_ggs(w, GGS_INTERVALS)

n_ggs = sum(w["is_ggs"] for w in words)
drop_frac = 1 - n_ggs / max(1, len(words))
print(f"[intersect] {n_ggs}/{len(words)} words inside GGS-only intervals "
      f"({100*n_ggs/max(1,len(words)):.1f}% kept; the rest are questioner/cross-talk).")

# HARD SANITY GATE: if the transcript is GGS-only (no questioner text), almost ALL words should land
# inside GGS intervals. A large drop means questioner text was aligned and smeared into GGS regions,
# dragging boundaries — investigate (set SPEAKER_LABEL_RE, or segment-align per GGS interval).
if drop_frac > MAX_WORD_DROP_FRAC:
    print(f"!! WARNING: {100*drop_frac:.1f}% of words dropped (> {100*MAX_WORD_DROP_FRAC:.0f}% gate). "
          "If transcript.txt is GGS-only this signals alignment smear from unlabeled questioner audio. "
          "Set SPEAKER_LABEL_RE/GGS_LABEL_RE (checklist item 3) or raise star_frequency, and re-run from Cell 8.")
else:
    print(f"[intersect] drop fraction {100*drop_frac:.1f}% within the {100*MAX_WORD_DROP_FRAC:.0f}% sanity gate.")

# Show a few dropped runs so you can confirm they're questioner turns, not GGS mis-drops.
dropped_runs, run = [], []
for w in words:
    if not w["is_ggs"]:
        run.append(w["text"])
    elif run:
        dropped_runs.append(" ".join(run)); run = []
if run: dropped_runs.append(" ".join(run))
print(f"[intersect] {len(dropped_runs)} dropped runs. First few (should read like questioner speech):")
for r in dropped_runs[:5]:
    print("   -", r[:120])

# %% Cell 10 — Stage E: group GGS-only words → clips + train.jsonl
# =====================================================================================
# Cell 10 — STAGE E: build sentence-boundary clips from GGS-ONLY contiguous word runs, cut
# 16k mono WAVs (both-end silence trim), write train.jsonl {audio,text,duration}.
# Clips are built PER GGS RUN so no clip ever straddles a questioner turn. Verses kept.
# Two fixes vs draft: (1) 'duration' = REAL cut-wav length (not the requested span, which over-counts
# trimmed silence); (2) the trailing pad is CLAMPED to the GGS-interval boundary so the clip can never
# bleed into the following (dropped) questioner turn.
# =====================================================================================
clips_dir = LOCAL_OUT / SLUG / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)
for old in clips_dir.glob("*.wav"):
    old.unlink()
manifest = LOCAL_OUT / SLUG / "train.jsonl"

def ggs_interval_end_after(t, intervals):
    """End of the GGS interval that contains time t (so we can clamp the tail pad to it)."""
    for s, e in intervals:
        if s - 1e-3 <= t <= e + 1e-3:
            return e
    return t   # not inside any interval (shouldn't happen for a GGS word) -> no extra tail

# Split the aligned words into contiguous GGS-only runs (questioner words break a run).
ggs_runs, cur = [], []
for w in words:
    if w["is_ggs"]:
        cur.append(w)
    elif cur:
        ggs_runs.append(cur); cur = []
if cur: ggs_runs.append(cur)

def emit_run(run_words, start_index, intervals):
    """Group ONE GGS run into sentence-boundary clips; cut + return manifest rows."""
    rows = []
    pairs = build_clips(run_words, MIN_SEC, MAX_SEC)     # indices are LOCAL to run_words
    for j, (s, e) in enumerate(pairs):
        idx = start_index + j
        prev_end = run_words[s - 1]["end"] if s > 0 else max(0.0, run_words[s]["start"] - 1.0)
        nxt = run_words[e + 1]["start"] if e + 1 < len(run_words) else run_words[e]["end"] + 1
        start = max(run_words[s]["start"] - LEAD_PAD, prev_end, 0.0)
        # Clamp the tail to (a) next GGS word onset, (b) TAIL_PAD, AND (c) the GGS-interval end —
        # so the clip never extends past GGS speech into the following questioner turn.
        iv_end = ggs_interval_end_after(run_words[e]["end"], intervals)
        end = run_words[e]["end"] + min(TAIL_PAD, max(0.0, nxt - run_words[e]["end"]))
        end = min(end, iv_end)
        txt = " ".join(run_words[k]["text"] for k in range(s, e + 1))
        dst = clips_dir / f"{SLUG}_{idx:04d}.wav"
        cut(wav16, start, end, dst)
        rows.append({"audio": f"clips/{dst.name}", "text": txt, "duration": round(wav_duration(dst), 2)})
    return rows

all_rows, next_idx = [], 1
for run_words in ggs_runs:
    rows = emit_run(run_words, next_idx, GGS_INTERVALS)
    all_rows.extend(rows); next_idx += len(rows)

with manifest.open("w", encoding="utf-8") as mf:
    for r in all_rows:
        mf.write(json.dumps(r, ensure_ascii=False) + "\n")

durs = [r["duration"] for r in all_rows]
if durs:
    print(f"[cut] {len(durs)} GGS-only clips ({sum(durs)/60:.1f} min, REAL wav durations)  "
          f"len: min={min(durs):.1f} avg={sum(durs)/len(durs):.1f} max={max(durs):.1f}s -> {clips_dir}")
print(f"[cut] manifest -> {manifest}")
print("NEXT: relisten to a few clips — coherent 2-3 sentence GGS ideas, no questioner, verses intact.")

# %% Cell 11 — Stage F: ASR round-trip CER/WER QA → train.clean.jsonl
# =====================================================================================
# Cell 11 — STAGE F: ASR ROUND-TRIP QA. Re-transcribe each clip (faster-whisper), score vs the
# clip's KNOWN text with jiwer. cer_core (Sanskrit-masked) drives pass/fail; cer_raw-cer_core =
# sanskrit_load is a DIAGNOSTIC (we NEVER toss a clip just for containing verses). Reuses qa.py's
# silence_edges audio check. Output: train.clean.jsonl + qa_asr_report.tsv + printed distributions.
# =====================================================================================
import numpy as np, librosa, jiwer
from faster_whisper import WhisperModel
from collections import Counter

SIL_DB = -40
def silence_edges(y, sr):                                  # verbatim from qa.py
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    voiced = np.where(db > SIL_DB)[0]
    if len(voiced) == 0:
        return None
    t = librosa.frames_to_time(voiced, sr=sr, hop_length=256)
    return float(t[0]), float(len(y) / sr - t[-1])

# Sanskrit/Vaishnava loanwords masked out of BOTH ref & hyp for cer_core (English backbone CER).
SANSKRIT_TOKENS = {
    "krishna","krsna","rama","hare","govinda","gopala","bhagavan","bhagavad","bhagavatam","shrimad",
    "srimad","gita","vedas","veda","vedanta","upanishad","bhakti","bhakta","jiva","jivatma","atma",
    "atman","paramatma","brahman","brahma","maya","moha","karma","jnana","yoga","yogi","guru","acarya",
    "acharya","prabhupada","chaitanya","caitanya","mahaprabhu","vaishnava","vaisnava","namabhasa","nama",
    "japa","kirtan","kirtana","mantra","prasada","prasadam","samadhi","sadhana","sadhu","shastra","sastra",
    "smriti","sruti","dharma","adharma","sankirtana","harinama","vrindavan","vrndavana","mathura","dvaraka",
    "goloka","vaikuntha","purusa","purusha","prakriti","prakrti","gunas","guna","sattva","rajas","tamas",
    "om","aum","namo","namah"}

_cer_tx = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                         jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                         jiwer.ReduceToListOfListOfChars()])
def _cer_tx_core():
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                          jiwer.RemoveSpecificWords(sorted(SANSKRIT_TOKENS)),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                          jiwer.ReduceToListOfListOfChars()])
_wer_tx = jiwer.wer_standardize

def score_text(ref, hyp):
    if not hyp.strip():
        return {"cer_raw":1.0,"cer_core":1.0,"wer":1.0,"sanskrit_load":0.0}
    cer_raw  = jiwer.cer(ref, hyp, reference_transform=_cer_tx,        hypothesis_transform=_cer_tx)
    cer_core = jiwer.cer(ref, hyp, reference_transform=_cer_tx_core(), hypothesis_transform=_cer_tx_core())
    try:
        wer = jiwer.wer(ref, hyp, reference_transform=_wer_tx, hypothesis_transform=_wer_tx)
    except ValueError:
        wer = 1.0
    return {"cer_raw":round(float(cer_raw),4),"cer_core":round(float(cer_core),4),
            "wer":round(float(wer),4),"sanskrit_load":round(float(cer_raw-cer_core),4)}

_compute = "float16" if DEVICE == "cuda" else "int8"
asr = WhisperModel(ASR_MODEL, device=DEVICE, compute_type=_compute)
def transcribe(wav):
    segs, _ = asr.transcribe(str(wav), language="en", beam_size=5, vad_filter=True,
                             vad_parameters=dict(min_silence_duration_ms=500),
                             condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs).strip()   # segs is a GENERATOR — must iterate

def run_qa(lec_root: Path, slug: str):
    """Score lec_root/<slug>/train.jsonl -> writes train.clean.jsonl + qa_asr_report.tsv; returns (rep,passed)."""
    d = lec_root / slug
    rows = [json.loads(l) for l in open(d / "train.jsonl", encoding="utf-8") if l.strip()]
    rep, passed = [], []
    print(f"[qa:{slug}] re-transcribing + scoring {len(rows)} clips ...")
    for i, r in enumerate(rows, 1):
        wav = d / Path(r["audio"]).as_posix(); reasons = []
        if not wav.exists():
            reasons.append("missing_wav"); m = {"cer_raw":1.0,"cer_core":1.0,"wer":1.0,"sanskrit_load":0.0}; lead=trail=-1.0
        else:
            m = score_text(r["text"], transcribe(wav))
            if m["cer_core"] > MAX_CER_CORE: reasons.append(f"high_cer_core({m['cer_core']:.2f})")
            if m["wer"]      > MAX_WER:      reasons.append(f"high_wer({m['wer']:.2f})")
            y, sr = librosa.load(str(wav), sr=16000); se = silence_edges(y, sr)
            if se is None:
                reasons.append("all_silence"); lead=trail=-1.0
            else:
                lead, trail = round(se[0],2), round(se[1],2)
                if se[0] > MAX_LEAD_SIL:  reasons.append(f"lead_sil({se[0]:.2f})")
                if se[1] > MAX_TRAIL_SIL: reasons.append(f"trail_sil({se[1]:.2f})")
        ok = not reasons
        rep.append({"clip":i,"dur":r.get("duration"),**m,"lead":lead,"trail":trail,
                    "status":"PASS" if ok else "FAIL","reasons":";".join(reasons)})
        if ok: passed.append(r)
        if i % 20 == 0 or i == len(rows): print(f"   {i}/{len(rows)} (pass so far: {len(passed)})")
    (d / "train.clean.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in passed), encoding="utf-8")
    cols = ["clip","dur","cer_raw","cer_core","wer","sanskrit_load","lead","trail","status","reasons"]
    with (d / "qa_asr_report.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for x in rep: f.write("\t".join(str(x[c]) for c in cols) + "\n")
    return rep, passed

def pctl(v, p): return round(float(np.percentile(v, p)), 3) if v else float("nan")

rep, passed = run_qa(LOCAL_OUT, SLUG)
nfail = sum(x["status"] == "FAIL" for x in rep)
c = Counter()
for x in rep:
    if x["status"] == "FAIL":
        for rs in x["reasons"].split(";"): c[rs.split("(")[0]] += 1
print(f"\n{len(rep)} clips -> PASS {len(passed)} / FAIL {nfail}  (max_cer_core={MAX_CER_CORE} max_wer={MAX_WER})")
print(f"fail reasons: {dict(c)}")
print("--- distributions (min / p10 / p50 / p90 / max) — RE-SET THRESHOLDS FROM THESE ---")
for name in ("cer_core","cer_raw","wer","sanskrit_load"):
    v = sorted(x[name] for x in rep)
    print(f"  {name:13s}: {pctl(v,0)} / {pctl(v,10)} / {pctl(v,50)} / {pctl(v,90)} / {pctl(v,100)}")
print("NOTE: high cer_core + LOW sanskrit_load = real bad clip. high cer_core + HIGH sanskrit_load =")
print("      just verses -> extend SANSKRIT_TOKENS or relax MAX_CER_CORE; do NOT drop the verse.")
print(f"clean manifest -> {LOCAL_OUT/SLUG/'train.clean.jsonl'} ({len(passed)} clips)")

# %% Cell 12 — Persist prototype outputs to Drive (batched)
# =====================================================================================
# Cell 12 — Sync local /content/out -> Drive in ONE batched rsync (avoids slow per-file FUSE writes).
# Full-lecture wavs live under /content/_work (NOT /content/out), so they are never uploaded to Drive.
# Run after the prototype looks good; outputs then survive a runtime reset.
# =====================================================================================
import subprocess
r = subprocess.run(["rsync", "-a", "--info=stats1", f"{LOCAL_OUT}/", f"{DRIVE_OUT}/"],
                   capture_output=True, text=True)
print(r.stdout or r.stderr)
print(f"persisted {LOCAL_OUT} -> {DRIVE_OUT} (clips + train.jsonl + train.clean.jsonl + report).")
print("INSPECT NOW: open a few clips under out/<slug>/clips/ and eyeball qa_asr_report.tsv before scaling.")

# %% [markdown]
# # ═══════════════ SCALE TO ALL LECTURES ═══════════════
# 
# Once the single-lecture prototype looks right (clips are GGS-only, verses intact, QA distributions sane), run the section below to loop **every** lecture under `lectures/`, then build the final `train.jsonl` + `val.jsonl` with a **whole-lecture GROUP split** (hold out entire lectures for val — never split a lecture's clips across train/val, or you leak recording conditions) and **ref_audio injection** into ~40% of train lines.
# 
# The loop **reuses the already-loaded models** (`diar_pipe`, `classifier`/`ggs_ref`, `alignment_model`, `asr`) — do not reload them per lecture. Each lecture produces its own `out/<slug>/train.clean.jsonl`; the split cell concatenates those.
# 
# **Per-lecture guardrail at scale:** the scale driver enforces the SAME speaker-verify gates as the prototype (low top-cosine, small margin to 2nd-best, cosine-vs-talk-time disagreement) and the SAME word-drop sanity gate. A lecture that trips any gate is **collected into `FLAGGED`** and printed at the end so you can review it before its clips enter the dataset — it is not silently anointed.

# %% Cell 13 — Run Stages A–F for every lecture (one function, looped)
# =====================================================================================
# Cell 13 — Per-lecture driver: runs Stages A-F end to end for ONE slug, writing
# out/<slug>/{clips, train.jsonl, train.clean.jsonl, qa_asr_report.tsv}. Then loop all lectures.
# Reuses the global models loaded above (diar_pipe, classifier/ggs_ref, alignment_model, asr).
# Enforces the SAME guardrails as the prototype (cosine threshold/margin/talk-time agreement,
# word-drop sanity gate) and collects FLAGGED lectures for human review instead of silently passing.
# =====================================================================================
FLAGGED = {}   # slug -> list of warning strings

def process_lecture(slug: str):
    print(f"\n========== {slug} ==========")
    flags = []
    ld = DRIVE_LECTURES / slug
    assert (ld / "audio.mp3").exists() and (ld / "transcript.txt").exists(), f"missing inputs for {slug}"
    w16 = to_wav_16k(ld / "audio.mp3", LOCAL_WORK / f"{slug}.wav")

    # A: diarize
    t, secs = diarize(diar_pipe, str(w16), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS, exclusive=True)
    # B: verify GGS cluster (reuse the global ggs_ref centroid built from your reference clips)
    sm, cn = defaultdict(float), defaultdict(int)
    for x in t:
        if (x["end"] - x["start"]) < MIN_TURN_SEC: continue
        sim = cosine(embed_wav(load_wav_16k_mono(str(w16), x["start"], x["end"])), ggs_ref)
        sm[x["speaker"]] += sim; cn[x["speaker"]] += 1
    scos = {s: sm[s]/cn[s] for s in cn}
    assert scos, f"{slug}: no cluster long enough to verify GGS"
    gspk = max(scos, key=scos.get)
    ranked = sorted(scos.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    print(f"  GGS cluster={gspk} cos={scos[gspk]:+.3f} margin={margin:+.3f} talk={secs.get(gspk,0)/60:.1f}min")
    for spk, c in sorted(scos.items(), key=lambda kv: -kv[1]):
        print(f"     {spk}: cos={c:+.3f} talk={secs.get(spk,0)/60:5.1f}min")
    if scos[gspk] < COSINE_THR:           flags.append(f"low_cosine({scos[gspk]:.3f}<{COSINE_THR})")
    if len(ranked) > 1 and margin < COSINE_MARGIN: flags.append(f"low_margin({margin:.3f}<{COSINE_MARGIN})")
    if max(secs, key=secs.get) != gspk:   flags.append("cosine!=talktime")
    # B-intervals (gap-bridge guarded against questioner-in-gap)
    intervals = build_ggs_intervals(t, gspk)
    # C: align KNOWN GGS-only transcript
    raw_txt = (ld / "transcript.txt").read_text(encoding="utf-8")
    txt = transliterate(ggs_only_text(raw_txt, SPEAKER_LABEL_RE, GGS_LABEL_RE).replace("\n", " "))
    wa = align_known_transcript(str(w16), txt, language=LANGUAGE, batch_size=8)
    ws = [{"text": x["word"], "start": x["start"], "end": x["end"], "score": x["score"]} for x in wa]
    # D: intersect (OVERLAP_FRAC=0.9) + word-drop sanity gate
    for x in ws: x["is_ggs"] = word_is_ggs(x, intervals)
    nkeep = sum(v["is_ggs"] for v in ws); dfrac = 1 - nkeep / max(1, len(ws))
    print(f"  words kept (GGS-only): {nkeep}/{len(ws)}  (drop {100*dfrac:.1f}%)")
    if dfrac > MAX_WORD_DROP_FRAC: flags.append(f"high_word_drop({dfrac:.2f}>{MAX_WORD_DROP_FRAC})")
    # E: clips + manifest (per GGS run; tail clamped to GGS interval; duration from real wav)
    cdir = LOCAL_OUT / slug / "clips"; cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.wav"): old.unlink()
    runs, cur = [], []
    for x in ws:
        if x["is_ggs"]: cur.append(x)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    rows, idx = [], 1
    for run_words in runs:
        for (s, e) in build_clips(run_words, MIN_SEC, MAX_SEC):
            prev_end = run_words[s-1]["end"] if s > 0 else max(0.0, run_words[s]["start"] - 1.0)
            nxt = run_words[e+1]["start"] if e+1 < len(run_words) else run_words[e]["end"] + 1
            start = max(run_words[s]["start"] - LEAD_PAD, prev_end, 0.0)
            iv_end = ggs_interval_end_after(run_words[e]["end"], intervals)
            end = run_words[e]["end"] + min(TAIL_PAD, max(0.0, nxt - run_words[e]["end"]))
            end = min(end, iv_end)
            ctext = " ".join(run_words[k]["text"] for k in range(s, e+1))
            dst = cdir / f"{slug}_{idx:04d}.wav"; cut(w16, start, end, dst)
            rows.append({"audio": f"clips/{dst.name}", "text": ctext, "duration": round(wav_duration(dst), 2)}); idx += 1
    with (LOCAL_OUT / slug / "train.jsonl").open("w", encoding="utf-8") as mf:
        for r in rows: mf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  cut {len(rows)} clips")
    # F: ASR QA
    rp, ps = run_qa(LOCAL_OUT, slug)
    print(f"  QA: PASS {len(ps)} / FAIL {sum(x['status']=='FAIL' for x in rp)}")
    if flags:
        FLAGGED[slug] = flags
        print(f"  !! FLAGGED for review: {flags}")
    return len(ps)

ALL_SLUGS = sorted(p.name for p in DRIVE_LECTURES.iterdir() if p.is_dir())
print("lectures to process:", ALL_SLUGS)
for s in ALL_SLUGS:
    if s == SLUG and (LOCAL_OUT / SLUG / "train.clean.jsonl").exists():
        print(f"\n========== {s} (already done in prototype, skipping) =========="); continue
    process_lecture(s)

# persist everything to Drive in one batched pass (clips live in /content/out; full wavs in /content/_work stay local)
import subprocess
subprocess.run(["rsync", "-a", "--info=stats1", f"{LOCAL_OUT}/", f"{DRIVE_OUT}/"], capture_output=True, text=True)
print("\nAll lectures processed + synced to Drive.")
if FLAGGED:
    print("\n!! REVIEW THESE FLAGGED LECTURES before trusting their clips:")
    for slug, fl in FLAGGED.items():
        print(f"   {slug}: {fl}")
else:
    print("No lectures flagged — speaker-verify + word-drop gates all passed.")

# %% Cell 14 — Whole-lecture GROUP split + ref_audio injection → train.jsonl / val.jsonl
# =====================================================================================
# Cell 14 — GROUP SPLIT + MANIFEST ASSEMBLY (pure stdlib). Hold out WHOLE lectures for val
# (no clip-level leakage). Namespace each clip path as <slug>/clips/<file> so names stay
# unique once merged. Inject ref_audio (another TRAIN clip, never self) into ~40% of TRAIN lines.
# =====================================================================================
import json, random
from pathlib import Path

def discover_lectures(root, manifest_name="train.clean.jsonl"):
    found = [(d.name, d / manifest_name) for d in sorted(p for p in Path(root).iterdir() if p.is_dir())
             if (d / manifest_name).exists()]
    if not found:
        raise FileNotFoundError(f"No '{manifest_name}' under {root}. Run Cell 13 first.")
    return found

def load_lecture_rows(slug, manifest_path, clips_subdir="clips"):
    rows = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        r["audio"] = f"{slug}/{clips_subdir}/{Path(r['audio']).name}"   # dataset-relative, unique
        r["lecture"] = slug                                            # group key (dropped on write)
        rows.append(r)
    return rows

def group_split(lectures_root, out_dir, manifest_name="train.clean.jsonl",
                val_lectures=None, n_val_holdout=1, ref_frac=0.40, seed=1337):
    rng = random.Random(seed)
    by_lec = {slug: load_lecture_rows(slug, m) for slug, m in discover_lectures(lectures_root, manifest_name)}
    lectures = sorted(by_lec)
    if len(lectures) < 2:
        print(f"!! Only {len(lectures)} lecture(s) — a real GROUP val needs >=2. Proceeding (val may be empty).")
    if val_lectures:
        val_lecs = set(val_lectures); assert not (val_lecs - set(lectures)), "val_lectures not found"
    else:
        k = min(n_val_holdout, max(0, len(lectures) - 1)) or (1 if lectures else 0)
        val_lecs = set(rng.sample(lectures, k)) if lectures else set()
    train_lecs = [l for l in lectures if l not in val_lecs]
    train = [r for l in train_lecs for r in by_lec[l]]
    val   = [r for l in val_lecs   for r in by_lec[l]]

    # ref_audio: only into TRAIN, drawn only from TRAIN clips, never self.
    train_audio = [r["audio"] for r in train]
    for i in (rng.sample(range(len(train)), int(round(ref_frac * len(train)))) if train else []):
        cand = train[i]["audio"]; ref = cand
        while ref == cand and len(train_audio) > 1:
            ref = rng.choice(train_audio)
        train[i]["ref_audio"] = ref

    assert set(r["lecture"] for r in train).isdisjoint(set(r["lecture"] for r in val)), "LEAKAGE: lecture in both"
    assert all(r.get("ref_audio") != r["audio"] for r in train), "ref_audio points at its own clip"

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    def _write(rs, path):
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k != "lecture"}, ensure_ascii=False) + "\n")
    _write(train, out_dir / "train.jsonl"); _write(val, out_dir / "val.jsonl")
    print(f"lectures           : {lectures}")
    print(f"VAL holdout (whole): {sorted(val_lecs)}")
    print(f"TRAIN lectures     : {train_lecs}")
    print(f"train.jsonl        : {len(train)} clips ({sum('ref_audio' in r for r in train)} with ref_audio ~{ref_frac:.0%})")
    print(f"val.jsonl          : {len(val)} clips")
    print(f"wrote -> {out_dir/'train.jsonl'} , {out_dir/'val.jsonl'}")
    return out_dir / "train.jsonl", out_dir / "val.jsonl"

# Build the final manifests from the per-lecture clean manifests living next to the clips on Drive.
# data_root for training = DRIVE_OUT (audio paths are '<slug>/clips/<file>.wav' relative to it).
group_split(lectures_root=DRIVE_OUT, out_dir=DRIVE_OUT,
            n_val_holdout=1,        # bump to 2 once you have >=8 lectures
            ref_frac=0.40, seed=1337)
print("\nNEXT: point VoxCPM's data_root at DRIVE_OUT (where <slug>/clips live), then run `voxcpm validate` "
      "on out/train.jsonl + out/val.jsonl before training.")

