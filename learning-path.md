# Learning path — beyond fastai, just-in-time for the GGS voice clone

**Principle (from [voice-cloning-research.md](voice-cloning-research.md), line 38):** *don't gate building
behind finishing courses.* Learn each piece right before the milestone that needs it. We ship the clone;
the learning rides along. Decision (2026-06-13): **focus on Tiers ①+② only, skip the rest.**

## The Sound-of-AI DSP course (local clone)

`../AudioSignalProcessingForML/` — Valerio Velardo's "Audio Signal Processing for ML" (23 lessons,
MIT). Slides = PDF per lesson; the implementation lessons have Jupyter notebooks (the actual code).
Videos: <https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0>.

We use ~10 of 23 lessons. Tiers ③ (Fourier math) and ④ (classification features) are **skip-for-now**.

## What we're doing — Tiers ①+②

### Tier ① — Core (the mel-spectrogram spine; do before the fine-tune)
- [ ] **L2** Sound & waveforms — sample rate, why 16 kHz
- [ ] **L4** Understanding audio signals — sampling / quantization / Nyquist
- [ ] **L15** Short-Time Fourier Transform — the transform behind spectrograms
- [ ] **L16** Extracting spectrograms (nb) — `stft → |S|² → power_to_db`
- [ ] **L17** Mel spectrogram explained — the mel scale, and *why*
- [ ] **L18** Extracting mel spectrograms (nb) — **the representation VoxCPM's vocoder predicts**

### Tier ② — Strong (maps straight onto our code; some due now at QA)
- [ ] **L3** Intensity / loudness / **timbre** — decibels, why we normalize; timbre = "voice character"
- [ ] **L7** Time-domain features — theory for L8/L9
- [ ] **L8** Amplitude envelope — onset / silence
- [ ] **L9** RMS energy + ZCR — **literally our `silence_edges()` gate** in [cloud/pod_dataprep.py](cloud/pod_dataprep.py)
- [ ] **L19** MFCCs explained — classic speaker features; the ancestor of ECAPA
- [ ] **L20** Extracting MFCCs (nb)

## Milestone anchoring (learn-just-before)

| Project milestone | Learn first | Status |
|---|---|---|
| **QA listen** (finalize gate policy) | L3 (dB) + **L9 (RMS/silence)** | ← we are here |
| **VoxCPM architecture** (before LoRA) | L15 → L16 → L17 → **L18** (mel) ; L19/L20 (speaker features) | pending |
| **LoRA fine-tune** | fastbook Ch5 (done) + voxcpm-finetune-guide | pending |
| **Eval** | ECAPA cosine similarity (already used in pipeline) | pending |

## Loop (our established habit)
Claude teaches a lesson source-verified (gentle pace, full depth) → tied to our actual code → then
document the concept to the Notion **ML Journey** DB via `/note`. Tick the box here when done.

## Skip-for-now (revisit only if a milestone pulls them in)
Tier ③ math: L10–L14 (Fourier intuition → complex numbers → DFT). Tier ④: L1, L5–L6 overviews,
L21–L23 (band-energy-ratio, spectral centroid/bandwidth — classification features, not TTS).
