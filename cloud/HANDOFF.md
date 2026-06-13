# HANDOFF — for a Claude Code session running on the Lightning Studio

You are continuing the **GGS voice-cloning** project on a cloud GPU. The user (Mukul) is a TS/Next.js SWE,
newer to ML, doing this as devotional seva (non-commercial). His laptop has no GPU, so the heavy work moved
here. The project *memory* lives on his laptop, not here — **this file + the repo docs are your source of truth.**

## Read these first (in order)
1. `cloud/HANDOFF.md` — this file.
2. `finetune-plan.md` — the full VoxCPM2 LoRA plan (clip recipe, manifest schema, LoRA config, eval, compute caveats).
3. `tts-dataprep-pipeline.md` — the field-standard data-prep pipeline + every verified decision and correction.
4. `colab/README.md` + `colab/ggs_dataprep_colab.py` — the **verified data-prep pipeline** (17 cells) to run here.
5. `scripts/` — the working LOCAL pipeline: `fetch_lecture.py`, `classify_transliterate.py` (transliterate util),
   `align_cut.py` (stable-ts align + clips), `qa.py` (QA). Reuse `transliterate()`, `build_clips()`, `cut()` verbatim.

## Where things stand
- **Voice baseline DONE:** VoxCPM2 zero-shot clone already run (Kaggle T4) — recognizably GGS.
- **Local data-prep WORKS:** lecture 1 → 131 clips, 126 pass QA. Scripts run on CPU. Only **1 of ~10 lectures fetched**.
- **Colab/cloud notebook BUILT + adversarially reviewed:** `colab/ggs_dataprep_colab.ipynb` (+ `.py`). Pipeline:
  diarize (pyannote) → ECAPA verify GGS cluster → **ctc-forced-aligner (MMS-300m)** on the known transcript →
  intersect word-timestamps with GGS-only intervals (drops questioners) → reuse our clip recipe → ASR CER/WER QA →
  whole-lecture group split + ref_audio. **Not yet run on a GPU.**

## Locked decisions (do NOT re-litigate)
- **KEEP recited Sanskrit verses** — GGS speaks (not chants) them; his real voice. No verse excision anywhere.
- **Transliterate IAST→ASCII with our custom lossy map** (`scripts/classify_transliterate.transliterate`).
  indic-transliteration was tested and rejected (reversible schemes; leaves ṁ non-ASCII).
- **Aligner = ctc-forced-aligner**, NOT WhisperX (WhisperX only aligns its own fresh transcription → would
  discard the accurate transcript). stable-ts is the cached fallback only.
- **Transcripts are GGS-only TEXT** (questioner turns generally untranscribed) → keep `SPEAKER_LABEL_RE=None`,
  rely on the Stage-D diarization geometry + the >15% word-drop sanity gate.

## Your next steps here (GPU)
1. **Env:** create a venv; install per `colab/` Cell 1 (the torch==2.8 + ctc-forced-aligner `--no-deps` ordering is
   load-bearing — read the cell comments). The ctc-aligner C++ build that failed on the user's Windows compiles fine here (Linux).
2. **Heads-up risks to verify on first run** (from the notebook read-through): the pyannote `community-1` API +
   `exclusive_speaker_diarization` (fallback: `speaker-diarization-3.1` + `.itertracks`); jiwer 4.0 arg names;
   make the Whisper cuDNN smoke test use the same `float16` as real QA.
3. **Need from the user:** an HF token + accept the gated pyannote repos; ONE clean GGS-only reference clip for ECAPA.
4. **Prototype on lecture 1**, eyeball clips + QA distributions, calibrate `COSINE_THR` / `MAX_CER_CORE`.
5. **Fetch the other ~9 lectures** (`scripts/fetch_lecture.py <url>`), scale, group-split, `voxcpm validate`.
6. **Train:** VoxCPM2 LoRA per `finetune-plan.md` (T4 16GB → VoxCPM1.5 or shrink batch; L4/24GB → VoxCPM2). Eval vs
   the zero-shot baseline with ECAPA cosine similarity.

## Working style the user expects
Gentle pace, full depth, frequent check-ins, wait for confirmation on big moves, source-verified over hand-rolled.
He documents ML concepts learned into a Notion "ML Journey" DB (he'll trigger that himself via `/note` or "document this").
