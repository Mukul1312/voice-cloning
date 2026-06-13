# Field-Standard TTS Data-Prep Pipeline (research-verified)

**Source:** deep-research pass (24/25 claims confirmed) + Firecrawl of the primary repos, June 2026. Full report: `tasks/wvdk0drg9.output`. This supersedes our hand-rolled heuristics where noted.

## The big realization
We were solving the wrong layer. **Stop detecting Sanskrit in TEXT** (our dict trick broke because loanwords like `krishna`, `bhakti`, `koruna` ARE English-dictionary words). The field drops **chanting in the AUDIO** (where prosody actually separates chant from speech) and keeps *all* the spoken text. That single shift dissolves the verse-leak bug at its source.

## The field-standard pipeline
`fetch → transliterate text → [audio QA: drop chant + keep-only-GGS] → forced-align known transcript → slice → QA-filter`

| Stage | Standard tool | Replaces our… | Install | Win/Colab |
|---|---|---|---|---|
| **Text: IAST→ASCII** | **our custom lossy IAST map** (`classify_transliterate.transliterate`) — **indic-transliteration TESTED & REJECTED** (its schemes are *reversible*: Kṛṣṇa→`kRSNa`/`kRShNa`, and they leave the anusvāra ṁ as a raw diacritic → wrong for a TTS frontend; our map gives the clean `Krishna`) | the dict-based Sanskrit **detection** (deleted); the map itself stays | (stdlib) | ✅ Windows (pure Py) |
| ~~Drop chant/sung~~ **N/A** | ~~inaSpeechSegmenter~~ — **TESTED, doesn't apply**: GGS *speaks* (recites) verses, doesn't melodically *chant* (probe: gliding/falling pitch, not sustained — `scripts/probe_chant.py`, `probe_verse.png`). No singing to detect. → **verse = product choice: KEEP (his voice, fine) or text-filter via language-ID** (IndicLID/lexicon), NOT acoustic. | retires text-based verse excision either way | — | skip |
| **Keep only GGS** | **pyannote.audio** diarization + **SpeechBrain ECAPA-TDNN** verify (cosine-sim vs a GGS reference embedding) | the **questioner leak** (#83) — nothing of ours did this | `pip install speechbrain pyannote.audio` (+ HF token) | ✅ Windows (PyTorch) |
| **Forced align** | **ctc-forced-aligner** (MMS-300m CTC) — **WhisperX REJECTED** (verified: it only aligns Whisper's *own* fresh transcription → would discard our accurate transcript). ctc-aligner aligns our KNOWN romanized transcript, never re-transcribes; also dodges torchaudio's removed forced-align ops. stable-ts kept only as a cached **fallback** (archived). | `align_cut`'s stable-ts aligner | `pip install git+…/ctc-forced-aligner` (Colab Linux — the MSVC build that failed on Windows compiles fine here) | 🚀 Colab |
| **Slice** | reuse our `build_clips` sentence-boundary grouping verbatim (clips only from GGS-only word runs) | — (kept) | — | — |
| **QA filter** | **ASR round-trip CER/WER** (re-transcribe each clip with Whisper, compare to its text) + edge-CER + silence/gap | `qa.py`'s **words-per-sec** proxy | (whisper already installed) | ✅ |

## Minimal usage (verified from the repos)
```python
# IAST -> readable ASCII: KEEP our own custom map (scripts/classify_transliterate.py).
# indic-transliteration was tested and REJECTED — its schemes are reversible romanizations
# (kRSNa / kRShNa) and leave ṁ as a raw diacritic; we want lossy clean ASCII ("Krishna").
from classify_transliterate import transliterate   # Kṛṣṇa -> Krishna, māyā-moha -> maya-moha

# Drop chant: inaSpeechSegmenter -> zones labeled speech / music(=singing) / noise
from inaSpeechSegmenter import Segmenter
seg = Segmenter(); zones = seg('audio.wav')   # keep 'speech' zones, drop 'music'

# Keep only GGS: ECAPA speaker embedding + cosine similarity (Ch8 idea!)
from speechbrain.inference.speaker import EncoderClassifier
clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
emb = clf.encode_batch(signal)   # cosine-compare each clip to a clean GGS reference; drop low-sim

# Forced align (WhisperX, replaces stable-ts which is archived)
import whisperx   # align() gives word timestamps; bundles VAD + pyannote diarization
```

## What we KEEP vs REPLACE
- **KEEP:** `fetch_lecture.py` (MP3 + transcript scrape, stdlib).
- **DONE (local quick-win, 2026-06-12):** `classify_transliterate.py` → **deleted** the dict-based Sanskrit detection (verses are kept, not detected); **KEPT** our custom IAST map (indic-transliteration tested & rejected) + added typographic-punctuation cleanup → output is now **100% ASCII**, zero non-ASCII on the real transcript.
- **DONE (local quick-win):** `align_cut.py` → **removed verse excision** (keep all words); pipeline is now transliterate → align → sentence-clips → cut. Re-ran lecture 1: **131 clips, 126 pass QA**, verse recitations retained intact (e.g. clip 0005). Deleted spent `probe_chant.py` (verdict captured here + `probe_verse.png`).
- **BUILT (Colab phase, 2026-06-12):** `colab/ggs_dataprep_colab.ipynb` (+ `.py` mirror + `README.md`) — a verification-driven 17-cell notebook (authored via an adversarially-reviewed workflow). Pipeline: diarize (pyannote 4.0 community-1) → ECAPA verify which cluster is GGS → **ctc-forced-aligner** on the known transcript → **intersect** word timestamps with GGS-only intervals (drops questioners by geometry) → reuse our `build_clips`/`cut` recipe → **ASR round-trip CER/WER QA** (Sanskrit-masked `cer_core` for pass/fail) → whole-lecture **group split** + `ref_audio` injection. Prototype-one-lecture-then-scale. **Not yet run** by the user (needs HF token + accept pyannote terms + GPU).
- Key verified call in that build: **WhisperX is wrong** for known-transcript alignment (aligns its own transcription) → use **ctc-forced-aligner**; torch must be pinned **2.8** (torchaudio 2.9 removes AudioMetaData → pyannote crash; torchcodec ≥0.8 segfaults).

## ⚠️ Honest caveats (verified, must test — don't copy numbers blind)
1. **inaSpeechSegmenter** is verified to tag *singing* as music, but was trained on French broadcast — whether it tags **melodic Sanskrit chant** as music is **unverified → empirical test on real GGS verses required.**
2. **ASR round-trip CER** will **mis-score Sanskrit loanwords** (Whisper won't transcribe them well) → calibrate thresholds on this data (published 30%/75%/60% are starting points, not constants).
3. **English-trained aligners may skip/mis-time IAST loanwords** (WhisperX drops tokens lacking dict chars) → transliterating to ASCII first helps; calibrate.
4. **Diarization is imperfect** + yields anonymous clusters → the ECAPA reference-verify step is required to pick GGS.
5. **stable-ts is archived** (read-only May 2026) → prefer WhisperX for anything long-term.

## Compute recommendation
Almost everything is **pip-installable pure-Python/PyTorch** (works on the Windows box — unlike the ctc-forced-aligner that needed a C++ compiler). BUT this is now ~5 model downloads (whisperx, pyannote, ECAPA, inaSpeechSegmenter, whisper-QA) on a slow CDN. Given that + GPU speed + training lives there anyway → **assemble and run this pipeline on Colab GPU.** Use the local Windows box only for light steps (fetch, transliterate).

## Open (research didn't fully resolve)
- Best aligner for heavily code-switched Sanskrit-English (NeMo CTC-Seg vs MFA+custom-dict vs MMS) — test.
- Whether VoxCPM2 needs a custom IAST G2P (we already decided: transliterate to ASCII — see `finetune-plan.md` D4).

## Rerun inputs (to regenerate this research)
deep-research workflow, brief = "field-standard tools for single-speaker TTS dataset prep from long-form code-switched lectures + transcripts" (6 angles: segmentation, chant removal, diarization, code-switch text, dataset QA, turnkey pipelines).
