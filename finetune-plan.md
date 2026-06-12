# VoxCPM2 LoRA Fine-Tune Plan — GGS English Voice

**Goal:** LoRA fine-tune **VoxCPM2** on clean English GGS clips to push speaker similarity above the zero-shot baseline. Devotional seva; Apache-2.0 (commercial-OK).

**Source of truth:** VoxCPM fine-tuning guide (scraped → `.firecrawl/voxcpm-finetune-guide.md`) and repo README (`.firecrawl/voxcpm.md`). All specs below are verified from those, June 2026.

**Current assets:** 10+ cleaned, fully-transcribed English lectures, 30+ min each (~5+ hours). Zero-shot baseline already run on Kaggle T4 (recognizably him; "Ultimate Cloning" closer).

---

## ⚠️ Compute caveat (decide at training phase)
VoxCPM2 LoRA ≈ **20 GB VRAM** at default `batch_size=16` / `max_batch_tokens=8192`. Free **T4 = 16 GB** → would OOM. Options:
1. Shrink `batch_size` (e.g. 4–8) + `max_batch_tokens` + raise `grad_accum_steps` to keep effective batch (single T4).
2. **Kaggle free 2×T4** via multi-GPU `torchrun --nproc_per_node=2` (~32 GB total).
3. Fall back to **VoxCPM 1.5 LoRA (~12 GB, fits T4)** — loses 48 kHz super-res + controllable cloning.
- VoxCPM 1.5 LoRA: ~12 GB · VoxCPM2 LoRA: ~20 GB · VoxCPM2 full FT: ~40 GB.

---

## Phase 1 — DATA (the decisive ~80%)

### Clip recipe (what to keep)
- **Length:** 3–30 s (sweet spot), target **~5–15 s**, cut at **sentence/phrase boundaries**. `<1s` unstable, `>30s` filtered. Never cut mid-word.
- **Count:** **~50–100 curated clips** to start (official single-speaker range is 5–50; 50–500 for richer style). DO NOT use all 5 h. Quality ≫ quantity.
- **KEEP:** clean **English only**; GGS solo (no audience/overlap); clear articulation; transcript matches audio exactly; **variety** of sentences/pace/emotion.
- **DROP:** any Odia/Bengali/Sanskrit code-switch; audible noise (model very noise-sensitive); kirtan/chanting; coughs/applause/long pauses; `<1s` or `>30s`.

### Per-clip preprocessing
- **Trim trailing silence < 0.5 s** (else "generation won't stop" after FT).
- Normalize volume across clips. Save **WAV**.
- No manual resample needed (dataloader resamples; config `sample_rate: 16000`).

### Source (UPDATED) — taptajivanam.com (official archive)
Each lecture page (e.g. `view-archive.php?...&a_id=N`) has a **direct MP3** in an `<audio><source src=".../custom_uploads/publications/audio/<file>.mp3">` tag + a **full human transcript** in the page body. `audio-archive.php` lists all lectures (enumerate to batch). This beats YouTube: the official transcript is accurate (guide demands exact text match). Supersedes the YouTube/Whisper trial (`scripts/prep_trial.py`).

### ⚠️ Code-switching filter (the key insight — refined after reading a real transcript)
Sanskrit is interleaved at the **phrase level**, not sentence level: almost every English sentence has embedded Sanskrit terms (Kṛṣṇa, māyā-moha), while full **verse recitations** are continuous Sanskrit runs that are *chanted* (different prosody). So the policy is NOT "pure English only":
- **KEEP** English-explanation clips even with sprinkled Sanskrit terms (= his real English voice).
- **DROP** only continuous **verse recitations** (chant prosody) + **questioner** turns.

### How to segment (per lecture) — TWO approaches
**Trial (robust, no fragile alignment) — `scripts/align_cut.py`:**
1. faster-whisper transcribe (lang=en, word timestamps, VAD).
2. Drop **low-confidence segments** (avg_logprob below threshold) — chanted Sanskrit / noise / unclear questioner.
3. Pack high-confidence English segments into 5–15 s clips; cut; trim trailing silence.
4. Text = Whisper's (good for clean English); manifest `train.jsonl`. Manual listen-review the ~15 clips.

**Production (RESEARCH-VERIFIED — deep-research pass, 19 claims confirmed):**
Pipeline order: parse GGS-only turns → **classify verse-vs-English on IAST diacritics** (+ `[verse]`/`[Song]` markers) → **transliterate IAST→ASCII** → **MFA-align the FULL ASCII text** to audio → **drop verse spans by timestamp** → cut English clips → manifest with ASCII text. (Align full text first so timestamps match audio; drop verses after.)

### Research-verified decisions
- **D4 Text = transliterate IAST→plain ASCII** (`Kṛṣṇa`→`Krishna`). VoxCPM's TEXT frontend is a MiniCPM-4 BPE tokenizer (Chinese+English-centric), NOT byte-level — IAST ingests but pronunciation is untested. VoxCPM maintainer (issue #47): limited data → LoRA + transliterate. CMUDict phoneme escape-hatch exists for stubborn words. [arXiv 2509.24650; VoxCPM#47]
- **D2 Aligner = MFA primary** (beats WhisperX/MMS at all thresholds, Interspeech 2024 arXiv 2406.19363; aligns ALL transcript words; Windows 1-line conda; `mfa find_oovs`→`mfa g2p` for OOV Sanskrit). **ctc-forced-aligner (MMS, Colab GPU) = fallback** (needs uroman romanization; default model CC-BY-NC).
- **Convergence:** ASCII transliteration serves BOTH model text AND aligner input — romanize once, reuse.

### Step 2a validation (DONE — `scripts/classify_transliterate.py`)
- **D4 (transliterate IAST→ASCII): ✅ validated** — custom map gives clean conventional ASCII (pumsah striya mithuni-bhavam, Rishabhadeva, Shrimad Bhagavatam, krishna-bhakti).
- **D3 (Sanskrit word detection): ✅ validated** — diacritics ALONE under-detect (many Sanskrit words lack diacritics: tvak, caiva, gandivam). Fix = **diacritic OR not-in-English-dictionary** (dwyl words_alpha, 370k, cached `data/english_words.txt`). Verses then show runs of 8–12 and drop correctly.
- **Confirmed: verse removal must be at the WORD level, not sentence level.** 13/245 KEEP sentences are "mixed" (English + embedded verse fragment, e.g. the gandivam verse) — sentence-level can't split them. → After MFA word-alignment, **excise contiguous Sanskrit-word runs (≥~3, the validated detector per-word) by timestamp; keep English spans**; sprinkled single terms (run 1–2) stay (his English voice). Tunable: raise excise-threshold or use citation-adjacency to keep spoken term-lists.

### Step 2b/2c (WORKING) — align → cut → QA, per lecture
Scripts (run in `.venv`): `fetch_lecture.py` (stdlib) → `align_cut.py` (stable-ts `tiny`, alignment cached to `words.json`) → `qa.py`.
- **align_cut.py**: transliterate → stable-ts word align → excise Sanskrit runs (>=3) → group into **sentence-boundary clips ~9–20 s** (coherent 2–3 sentence ideas) → lead-pad + **both-end silence trim** → `train.jsonl`. (ctc-forced-aligner abandoned: needs MSVC C++ build tools on Windows. stable-ts = pure-Python, no compiler.)
- **qa.py** (the scalable QA backbone — metrics for coverage, gallery for spot-check): scores every clip on **words/sec** (low = alignment-failure / text-audio mismatch) + **lead/trail silence**; drops failures; writes `train.clean.jsonl` (curated), `qa_report.tsv`, `qa_gallery.png`.
- **Calibrated threshold: `--min-wps 0.65`** (NOT 1.0 — GGS speaks slowly/repetitively, so legit clips run 0.7–1.0 wps; only <0.65 is truly broken). Lecture 1: 133 clips → **127 clean**.
- **2D viz works**: mel-spectrogram + pitch let Claude SEE cut quality / silence / speech-vs-chant (caught the 26 s/3-word broken clip). Structural checks = Claude via charts; perceptual (sounds-like-GGS) = user's ears.

### TODO before scaling to all 10 lectures
- [ ] **Questioner filter (D5)**: `fetch_lecture.py` strips `Devotee:` / `Gour Govinda Swami:` labels — re-fetch KEEPING them, keep only GGS turns (a questioner clip leaked: #83). 
- [ ] **Group split**: hold out whole lectures for valid (not random clips) — see [[group-based-splitting-avoid-leakage]].
- [ ] Manifest `duration` currently = word-span (pre-trim); optionally recompute from actual wav.
- [ ] User listens to a spread of the 127 cleaned clips (perceptual gate).

### Empirical tests (proven vs assumed)
- **T1 (local):** run MFA on 2–3 lectures, spot-check boundaries by ear — MFA's clean-English benchmark win on Indic-accented code-switched audio is UNPROVEN.
- **T2 (GPU/Colab):** base-model A/B `māyā-moha` vs `maya-moha` pronunciation; default ASCII.
- Full report: `tasks/whwih2bmm.output` (temp; key facts captured here).

### Manifest (JSONL, one line per clip) — the deliverable of Phase 1
```json
{"audio": "clips/ggs_0001.wav", "text": "Exact transcript of this clip."}
{"audio": "clips/ggs_0002.wav", "text": "...", "ref_audio": "clips/ggs_0123.wav"}
```
- Required: `audio`, `text`. Optional: `ref_audio`, `duration`, `dataset_id`.
- Add `ref_audio` (another random GGS clip) to **30–50%** of lines → retains zero-shot + reference cloning.
- Split into `train.jsonl` + small `val.jsonl`.
- Run `voxcpm validate` (pre-flight data checker) before training.

---

## Phase 2 — TRAIN (LoRA, ~20%)
`pip install voxcpm` (Python ≥3.10 <3.13, PyTorch ≥2.5, CUDA ≥12). Download VoxCPM2 weights.

LoRA config (`conf/voxcpm_v2/voxcpm_finetune_lora.yaml`-style), verified defaults:
```yaml
pretrained_path: /path/to/VoxCPM2/
train_manifest: /path/to/train.jsonl
val_manifest:   /path/to/val.jsonl
sample_rate: 16000        # AudioVAE encoder input rate (NOT the 48kHz output)
out_sample_rate: 48000
batch_size: 16            # ↓ shrink for 16GB T4
grad_accum_steps: 1       # ↑ raise to compensate
num_iters: 1000
learning_rate: 0.0001
weight_decay: 0.01        # ← Ch8 weight decay
warmup_steps: 100
max_batch_tokens: 8192    # ↓ shrink for 16GB
lora: { enable_lm: true, enable_dit: true, enable_proj: false, r: 32, alpha: 32, dropout: 0.0 }
```
- `enable_dit: true` = essential for voice quality. `r=32` for speaker cloning (LoRA r=32 ≈ 98% of full-FT similarity); `r=48–64` if style/prosody under-captured.
- Launch: `python scripts/train_voxcpm_finetune.py --config_path <yaml>` (or `lora_ft_webui.py` GUI).
- **Stop early:** 1–3 epochs usually enough; **overfitting emerges very early** (Ch7/Ch8). If it starts ignoring text (same audio regardless of input) → overfit → roll back to earlier checkpoint.
- Monitor (TensorBoard): `loss/diff` (↓ then flatten), `loss/stop` (low/stable), `grad_norm` (spikes = bad samples / LR too high). Save multiple checkpoints near convergence; pick by ear.

---

## Phase 3 — EVAL
- Generate English test sentences with the LoRA'd model (try `r` / checkpoint variants).
- **Blind listen** test: a devotee who knows his voice.
- **Quantitative:** cosine similarity of speaker embeddings, real-GGS vs clone (Ch8 metric; embedder = ECAPA-TDNN). Compare LoRA vs zero-shot baseline to confirm improvement.

---

## Open decisions / TODO
- [ ] Pick segmentation tool (WhisperX vs existing timestamps) and cut first lecture → ~10 trial clips.
- [ ] Confirm compute path (single T4 shrunk batch / 2×T4 / VoxCPM1.5).
- [ ] Build `train.jsonl` + `val.jsonl`; run `voxcpm validate`.
- [ ] First LoRA run (r=32, 1–3 epochs); eval; iterate.
