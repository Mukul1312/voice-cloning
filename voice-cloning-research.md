# Voice Cloning Research — GGS English Voice

**Goal:** Clone the English voice of Gour Govinda Swami Maharaj (GGS) — whose speech is Indian/Odia-accented and code-switches into Odia, Bengali, and Sanskrit, recorded in old/noisy/low-quality audio, highly expressive. Devotional service (seva); open-source / self-hostable preferred. English first; Odia→Hindi later.

**Learner:** Software engineer (TypeScript / Next.js, GCP/Vertex background), new to ML/DL, at fast.ai Lesson 3.

**Provenance:** All tool facts below were verified live (June 2026) by scraping each model's own GitHub/HF page with Firecrawl — not from training knowledge. Raw scrapes cached in `.firecrawl/`. Sources linked throughout.

---

## PART 1 — Learning Path

### The reframing principle
> Voice cloning one speaker = **fine-tuning / few-shot conditioning of a large pretrained model**, never training from scratch. This is exactly fast.ai's core thesis (pretrained → fine-tune on small data).

### Coverage gap
| Resource | Gives you | On-target for voice cloning |
|---|---|---|
| **fast.ai** | DL mindset, training loop, transfer learning, embeddings, HF Transformers, data discipline | ~25–35% (zero audio content) |
| **Sound of AI** (musikalkemist) | Audio DSP, mel-spectrograms, DL-on-audio, first speech generation (VAE) | ~50–60% (stops before modern TTS) |
| **Modern TTS stack** | The actual cloning machinery | Neither course covers — learn at the repos |

### fast.ai — what to prioritize (from Lesson 3)
Do, then stop: **L3 (training loop, you're here) → L4 (NLP/HF Transformers — most relevant) → L5 + L7 (embeddings = speaker embeddings) → L2's data-cleaning + HF Spaces half.** Skim L1, L6, L8. **Skip the Part 2 Stable-Diffusion full build** (skim its concepts only).

### Sound of AI (musikalkemist) — order for voice cloning
1. **AudioSignalProcessingForML** — STFT, mel-spectrogram, MFCC via librosa (start early, even alongside fastai). Highest-leverage audio literacy.
2. **DeepLearningForAudioWithPython** (use the **v2 branch**) — NN→CNN→RNN on audio.
3. **pytorchforaudio** — switch to PyTorch + torchaudio (the cloning ecosystem is all PyTorch).
4. **generating-sound-with-neural-networks** (VAE on Free Spoken Digit Dataset) — first speech generation; teaches the spectrogram→waveform (vocoder) problem.
5. Skim **generativemusicaicourse** (VAE/GAN/Transformer lectures only). Skip the melody-RNN repo.
- Bonus repo found live: **tts-voicecloning-course** ("Monster TTS & Voice Cloning Course") — most directly on-topic.

### Stack neither course teaches (finish externally)
Speaker embeddings (ECAPA-TDNN/x-vectors), modern TTS (below), vocoders (HiFi-GAN/BigVGAN), dirty-audio preprocessing (denoise/VAD/forced-alignment), self-supervised speech (wav2vec2/HuBERT/WavLM), multilingual G2P/phonemizer for Odia/Bengali/Sanskrit.

### Timeline
~6–10 weeks focused study to be productive — but **start the hands-on baseline in week 1**; don't gate building behind finishing courses.

---

## PART 2 — Tool Landscape (verified June 2026)

### Two layers (don't conflate)
- **Voice identity** (sound like GGS, zero-shot from his reference): VoxCPM2 / IndexTTS-2 / Chatterbox / Fish / F5-TTS / XTTS-v2 / OpenVoice.
- **Indian-language phonetics** (Odia/Bengali/Sanskrit): Indic Parler-TTS / Sarvam. (These are TTS, **not** GGS-cloners — fine-tune to add him.)

For English-first, the **voice-identity layer matters now**.

### Verified shortlist
| Tool | License | Clone | Langs | Notes for GGS | Source |
|---|---|---|---|---|---|
| **VoxCPM2** ⭐ | **Apache-2.0 (commercial-OK)** | ✅ controllable + "ultimate" | 30 (incl Hindi; no Odia/Bengali/Sanskrit) | **16kHz ref → 48kHz out (built-in super-resolution)**; steer emotion/pace/style; context-aware prosody; **LoRA/SFT fine-tune**; ~real-time on RTX 4090 | [VoxCPM](https://github.com/OpenBMB/VoxCPM) |
| **IndexTTS-2** | verify | ✅ zero-shot | ZH/EN | **Emotion ⟂ timbre disentanglement + duration control** — best pure "bhava + pauses" control | [index-tts](https://github.com/index-tts/index-tts) |
| **Chatterbox** | **MIT** | ✅ zero-shot | EN (+23 multiling., incl Hindi) | `exaggeration` + `[laugh]`/`[cough]` tags; **every output watermarked (PerTh)**; simplest to run | [chatterbox](https://github.com/resemble-ai/chatterbox) |
| **Fish Audio S2 Pro** | flagship **cloud/commercial**; open weights non-commercial ⚠️ | ✅ 10–30s ref | 80+ | **sub-word emotion tags `[whisper][excited][angry]`**; SOTA naturalness/paralinguistics | [fish-speech](https://github.com/fishaudio/fish-speech) |
| **F5-TTS** | code permissive; **base checkpoint (Emilia) non-commercial** ⚠️ | ✅ zero-shot | EN/ZH base | explicit training/fine-tune UI; fast; very active | [F5-TTS](https://github.com/SWivid/F5-TTS) |
| **XTTS-v2** | ⚠️ **Coqui CPML = non-commercial** | ✅ 6-sec clip | 17 (incl Hindi; no Odia/Bengali/Sanskrit), 24kHz | most-documented fine-tune path (66 public finetunes); license risk for public platform | [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| **OpenVoice** | **MIT** | ✅ cross-lingual | EN/ES/FR/ZH/JA/KO | control of emotion/accent/rhythm/**pauses**/intonation; lightweight, lower fidelity | [OpenVoice](https://github.com/myshell-ai/OpenVoice) |
| **CosyVoice 3.0** | Apache-2.0 (verify) | ✅ zero-shot | 9 + 18 ZH dialects; **no Indian langs** | instruct control (emotion/speed); English not its focus | [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |

### Indian-language layer (for later Odia→Hindi)
| Tool | License | Notes | Source |
|---|---|---|---|
| **Indic Parler-TTS** | **Apache-2.0** | 21 langs incl **Odia, Bengali, Sanskrit, Indian-English**; NSS Sanskrit 99.8% / Odia 88.9%; expressivity/pitch/pause/noise/reverb controls; **NOT zero-shot clone** (69 preset/described voices) — **fine-tune to add GGS** | [indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) |
| **Sarvam Bulbul V3** | **cloud API (paid)** | 11 Indian langs incl **Odia + Bengali** (no Sanskrit); 48kHz; emotion; **native Hinglish code-switching**; consent-based cloning | [sarvam](https://www.sarvam.ai/text-to-speech) |

### Denoise / restore front-end (for GGS's degraded audio)
| Tool | License | Notes | Source |
|---|---|---|---|
| **Resemble Enhance** | MIT | denoise **+ bandwidth extension (super-resolution)** — restores distortions, not just noise; 44.1kHz; `pip install resemble-enhance` | [resemble-enhance](https://github.com/resemble-ai/resemble-enhance) |
| **DeepFilterNet** | MIT/Apache-2.0 | real-time full-band 48kHz noise suppression; lighter | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) |

### Recommended stack
```
GGS raw audio (noisy, low sample-rate)
  → [1] Resemble Enhance (denoise + bandwidth-extend)        MIT
  → [2] VoxCPM2 zero-shot clone (16kHz in → 48kHz out)       Apache-2.0, commercial-OK   ★ START
        compare vs IndexTTS-2 (bhava/pauses) · Chatterbox (simplest) · Fish S2 (emotion tags)
  → [3] if not close enough → LoRA fine-tune VoxCPM2 on clean English segments
  → (LATER, Odia→Hindi) fine-tune Indic Parler-TTS, or Sarvam (cloud)
```

### Why VoxCPM2 is the top pick
Only top-tier model that solves three constraints at once: (1) **Apache-2.0 commercial-ready** (clean for a public devotional platform — XTTS & F5 checkpoints are non-commercial); (2) **16kHz→48kHz super-resolution** built in (designed to upgrade low-quality input — aimed straight at GGS's bad recordings); (3) **controllable cloning + LoRA fine-tuning** (capture his timbre and steer bhava). Gap: Hindi but no Odia/Bengali/Sanskrit (fine for English-first).

### Prior-advice scorecard (corrected against sources)
- **XTTS-v2 go-to:** ⚠️ works, but **non-commercial license** (never flagged) and **no Odia/Bengali/Sanskrit**.
- **Chatterbox MIT + emotion:** ✅ confirmed; new fact — **outputs watermarked**; no Odia/Bengali/Sanskrit.
- **VoxCPM mixed-language:** ✅ confirmed and stronger — commercial Apache-2.0, super-resolution, LoRA.
- **Fish S2 noisy + emotion tags:** ✅ emotion confirmed; ⚠️ best model is cloud/non-commercial.
- **Sooktam-2 / Svara-TTS Indian:** 🔶 unverified; the verified Indian model is **Indic Parler-TTS** — and it's TTS, not a GGS-cloner.

### Honest hard parts
- **Mid-sentence code-switching** (Eng→Odia→Sanskrit): still the rough edge — synthesize clean English only for now.
- **Bhava:** more reachable than the pessimistic take — IndexTTS-2 emotion control + VoxCPM2 fine-tuning get close; the listener's spiritual *perception* is not a model property.

### Compute & licensing
- All need an **NVIDIA GPU** (or Colab/cloud); fine-tuning needs more VRAM than inference. VoxCPM2 ~real-time on RTX 4090.
- **Clean licenses for a public platform:** VoxCPM2 (Apache-2.0), Chatterbox (MIT), OpenVoice (MIT), Indic Parler-TTS (Apache-2.0), the denoisers (MIT/Apache). **Watch:** XTTS-v2 (CPML non-commercial), F5-TTS base checkpoint (Emilia non-commercial), Fish S2 Pro (cloud/commercial).

---

## START HERE (this week, zero theory)
1. `pip install resemble-enhance` → clean your best 5–10 min of GGS English.
2. Run **VoxCPM2** zero-shot with the cleaned clip as reference; generate 5–6 English test sentences.
3. Have a devotee who knows his voice listen blind. Timbre present but flat? → try **IndexTTS-2**, then **LoRA fine-tune VoxCPM2**.

(Easiest environment: Google Colab free GPU, or a local NVIDIA GPU.)
