# GGS TTS Data-Prep — Colab phase

Open **`ggs_dataprep_colab.ipynb`** in Google Colab on a **GPU** runtime. It force-aligns each
lecture's accurate human transcript to audio, keeps **only GGS** (drops questioner turns via
diarization + ECAPA speaker-verification), cuts sentence-boundary clips, and runs an ASR
round-trip CER/WER QA gate. **Verses are kept** (no excision anywhere).

Prototype on ONE lecture (Cells 1–12), eyeball it, then run SCALE (Cells 13–14) over all lectures
with a whole-lecture **group split**.

## Why ctc-forced-aligner (not WhisperX)
WhisperX only aligns Whisper's *own* fresh transcription — feeding it our transcript means
re-transcribing and discarding the accurate text (incl. romanized Sanskrit). ctc-forced-aligner
(MMS-300m CTC) aligns *our known transcript* to the audio and never re-transcribes. It also sidesteps
torchaudio's deprecated/removed forced-align ops (gone in torchaudio 2.9). stable-ts stays only as a
cached offline fallback. (The C++ build that failed locally on MSVC compiles fine on Colab's Linux.)

## Setup (do once, by hand)

1. GPU runtime: Runtime > Change runtime type > GPU (T4 is enough). Cell 2 asserts this.
2. Create a Hugging Face READ token at https://huggingface.co/settings/tokens and store it as a Colab secret named HF_TOKEN (left sidebar key icon, toggle notebook access).
3. While logged in to Hugging Face, open the community-1 model card (https://huggingface.co/pyannote/speaker-diarization-community-1) and ACCEPT THE TERMS FOR EVERY GATED REPO IT LISTS AS A REQUIREMENT (the card enumerates them — do not guess a specific segmentation repo). If Pipeline.from_pretrained returns None, you missed one.
4. Mount Drive and upload inputs: each lecture under MyDrive/ggs-voice-clone/lectures/<slug>/ containing audio.mp3 and transcript.txt.
5. TRANSCRIPT CONTRACT: ensure transcript.txt is GGS-only OR keeps speaker labels. The old fetch_lecture.py stripped 'Gour Govinda Swami:' / 'Devotee:' labels — re-fetch KEEPING them (a questioner clip leaked once: #83). If labels are present, set SPEAKER_LABEL_RE and GGS_LABEL_RE in the CONFIG cell so questioner TEXT is removed before alignment. If already GGS-only, leave them None and rely on the Stage-D geometry + the >15% word-drop sanity gate.
6. Designate a clean GGS-only reference clip (a few seconds, no questioner/music/applause), e.g. MyDrive/ggs-voice-clone/refs/ggs_ref_01.wav, and set REF_GGS_CLIPS in CONFIG. Add 2-3 clips spanning different lectures/rooms for a more robust centroid before scaling.
7. Spell out digits in transcripts (e.g. 1882 -> eighteen eighty two); uroman/MMS strips bare digits and desyncs alignment. The CONFIG cell prints any digit-bearing lines to fix.
8. Run order: Cell 1 (install) -> Runtime > Restart session -> Cell 2 (GPU check) -> Cell 3 (verify imports + Whisper cuDNN smoke test) -> Cell 4 (mount Drive + CONFIG) -> Cells 5-12 (prototype, inspect) -> Cells 13-14 (scale + group split). Do NOT re-run Cell 1 after the restart, and never pip install -U torch/torchcodec.
9. First runs download model weights (MMS-300m ~1.2GB, faster-whisper large-v3 ~3GB, ECAPA ~80MB) into the session cache; budget a few minutes.

## Caveats / things to calibrate

- Speaker-verify COSINE_THR=0.40, COSINE_MARGIN=0.10, and OVERLAP_FRAC=0.90 are starting points. Calibrate COSINE_THR/MARGIN on a handful of hand-labeled GGS vs questioner turns; the scale loop only FLAGS low-cosine/low-margin lectures, it does not auto-fix them — review FLAGGED lectures before their clips enter the dataset.
- ASR-QA thresholds (MAX_CER_CORE=0.30, MAX_WER=0.60, MAX_LEAD_SIL=0.6, MAX_TRAIL_SIL=1.0) are published starting points. Read the printed cer_core/cer_raw/wer/sanskrit_load distributions after the first lecture and re-set them; high cer_core with HIGH sanskrit_load is just verses (extend SANSKRIT_TOKENS, do not drop the clip).
- Transcript questioner-text risk: if transcript.txt still contains UNLABELED questioner turns (SPEAKER_LABEL_RE=None and it is not actually GGS-only), the aligner forces that text onto questioner audio and can drag adjacent GGS word boundaries. Stage D drops those words by geometry and hard-warns if >15% drop, but the clean fix is to keep speaker labels (re-fetch) or segment-align each GGS interval's audio against its transcript slice. Treat a tripped word-drop gate as a real data issue, not noise.
- The ctc-forced-aligner install COMPILES a pybind11 C++ extension from source (gcc on Colab GPU images, ~1-2 min). It is the slowest/most failure-prone install step; if main ever breaks the build, pin ALIGNER_SHA to a known-good commit (spec cites last-good 2026-04-15). It is installed with --no-deps so it can never float torch/torchcodec.
- Version pinning is load-bearing: torch==2.8.0 + torchaudio==2.8.0 + torchcodec==0.7 is the non-segfaulting set for pyannote 4.0. torchaudio>=2.9 removes AudioMetaData (pyannote crash) and torchcodec>=0.8 re-introduces the std::bad_alloc segfault. Cell 1 re-pins the trio LAST and Cell 3 asserts torchcodec==0.7; a stray pip -U of any of these silently breaks the runtime after restart.
- cuDNN-9 for ctranslate2/faster-whisper: LD_LIBRARY_PATH is set in Cell 2 (early, before any WhisperModel import) and Cell 3 runs a tiny CUDA WhisperModel smoke test so a 'libcudnn_ops.so.9' load failure surfaces in seconds. If the smoke test fails, restart and re-run Cell 2 then Cell 3, or fall back to compute_type='int8_float16'.
- Manifest 'duration' is now read from the actual trimmed wav (librosa.get_duration), so it matches what VoxCPM sees. This adds a small per-clip file read; negligible vs ASR/alignment cost.
- GROUP split needs >=2 lectures for a non-empty val. Keep n_val_holdout=1 until you have ~8 lectures, then bump to 2. ref_audio is injected into ~40% of TRAIN lines only, drawn only from TRAIN clips and never self.
- Whisper VAD (vad_filter=True, min_silence_duration_ms=500) can occasionally drop a quiet-but-valid clip to empty -> CER 1.0 -> false FAIL. If you see all-silence-ish PASS clips failing on text, try vad_filter=False and lean on the librosa silence_edges check instead.

## Note on our transcripts
Our taptajivanam transcripts are GGS-only *text* (questioner turns generally aren't transcribed),
so leave `SPEAKER_LABEL_RE=None` and rely on the Stage-D diarization geometry + the >15% word-drop
sanity gate. If a lecture trips that gate, its audio likely has substantial untranscribed questioner
speech — review it (it lands in `FLAGGED`).

