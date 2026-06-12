ContentsMenuExpandLight modeDark modeAuto light/dark, in light modeAuto light/dark, in dark mode[Skip to content](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html#furo-main-content)

[Back to top](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html#)

[View this page](https://voxcpm.readthedocs.io/en/latest/_sources/finetuning/finetune.rst.txt "View this page")

# Fine-Tuning Guide [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#fine-tuning-guide "Link to this heading")

This guide covers how to fine-tune VoxCPM with two approaches: LoRA (parameter-efficient) and full fine-tuning. Both use the same training script and data format.

* * *

## Environment & Resources [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#environment-resources "Link to this heading")

### Software [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#software "Link to this heading")

| Dependency | Version |
| --- | --- |
| Python | 3.10–3.11 recommended for training |
| PyTorch | 2.5.0+ |
| CUDA | 12.0+ |
| safetensors | recommended (falls back to `.bin` / `.ckpt` if unavailable) |

Additional Python packages used by the training script: `tensorboardX`, `argbind`, `transformers` (for the cosine scheduler), `librosa` (for validation mel spectrograms).

### Hardware [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#hardware "Link to this heading")

| Setup | LoRA | Full Fine-Tuning |
| --- | --- | --- |
| VoxCPM 1.5 (750M) | ~12 GB VRAM | ~24 GB VRAM |
| VoxCPM 2 (2B) | ~20 GB VRAM | ~40 GB VRAM |

These are rough estimates with `batch_size=16` and `max_batch_tokens=8192`. Actual usage depends on audio length and accumulation steps. If you hit OOM, see [Fine-Tuning FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html).

Multi-GPU training is supported via `torchrun`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path your_config.yaml
```

Copy to clipboard

* * *

## Data Preparation [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#data-preparation "Link to this heading")

### Format [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#format "Link to this heading")

Training data is a JSONL manifest file with one sample per line:

```
{"audio": "path/to/audio1.wav", "text": "Transcript of audio 1."}
{"audio": "path/to/audio2.wav", "text": "Transcript of audio 2.", "ref_audio": "path/to/audio1.wav"}
{"audio": "path/to/audio3.wav", "text": "Optional fields.", "duration": 3.5, "dataset_id": 1}
```

Copy to clipboard

| Field | Required | Description |
| --- | --- | --- |
| `audio` | Yes | Path to audio file (WAV recommended) |
| `text` | Yes | Transcript matching the audio content |
| `ref_audio` | No | Path to a reference audio clip from the **same speaker**. It is used as speaker-conditioning context for voice cloning, so it does not need to be an unseen sample. In practice, `ref_audio` is typically another clip randomly sampled from the same speaker / timbre as the target audio. When present, the training sequence is constructed as `[103, ref_feats, 104, text, 101, audio_feats, 102]`, teaching the model to clone the speaker’s voice from the reference. Loss is only computed on the target audio segment. |
| `duration` | No | Duration in seconds; speeds up length filtering |
| `dataset_id` | No | Integer ID for multi-dataset mixing (default: 0) |

See `examples/train_data_example.jsonl` in the repository for a reference.

Tip

**Mixing ref\_audio and non-ref\_audio samples** — We recommend that 30–50% of your training samples include `ref_audio`, so the model retains both zero-shot and reference-based voice cloning abilities. A simple strategy is to randomly choose another clean recording from the same speaker as `ref_audio` for each target sample.

### Audio requirements [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#audio-requirements "Link to this heading")

- **Format:** WAV is recommended. Other formats supported by torchaudio also work.

- **Sample rate:** The dataloader automatically resamples to the target model’s rate, so you do not need to pre-resample. The `sample_rate` in your training config must match the AudioVAE **encoder** input rate:

  - VoxCPM 1.0: 16kHz

  - VoxCPM 1.5: 44.1kHz

  - VoxCPM 2: 16kHz (the encoder operates at 16kHz; the decoder outputs 48kHz)
- **Duration:** 3–30 seconds per clip is the practical sweet spot. Very short clips (< 1s) produce unstable results. Very long clips increase VRAM usage and may be filtered out by `max_batch_tokens`.


### Preprocessing tips [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#preprocessing-tips "Link to this heading")

- **Trim trailing silence** to < 0.5 seconds. Long trailing silence is one of the most common causes of “generation doesn’t stop” after fine-tuning.

- **Normalize volume** if your recordings have inconsistent levels.

- **Clean transcripts:** Ensure the text matches the audio exactly. Mismatched text degrades both cloning quality and text adherence.

- **Remove noisy samples.** The model is sensitive to background noise in training data.


### Choosing your path [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#choosing-your-path "Link to this heading")

Your data size and goal determine which fine-tuning approach to use:

| Goal | Data Size | Recommended Approach |
| --- | --- | --- |
| Clone a single speaker | 5–50 clips | [LoRA Fine-Tuning](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html#lora-finetuning) — fast, low VRAM |
| Adapt to a domain or style | 50–500 clips | [LoRA Fine-Tuning](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html#lora-finetuning) — with higher rank (`r=32–64`) |
| Add a new language | 500+ hours | [Full Fine-Tuning](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html#full-finetuning) — mix with some Chinese/English data to reduce forgetting |
| Large-scale customization | 1000+ clips | [Full Fine-Tuning](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html#full-finetuning) |

**LoRA vs Full Fine-Tuning at a glance:**

In internal benchmarks on single-speaker cloning, LoRA (`r=32`) achieved approximately 98% of the speaker similarity of full fine-tuning, while using roughly half the VRAM and producing checkpoint files that are orders of magnitude smaller. LoRA is the recommended starting point for most tasks. Results may vary with different datasets and goals.

* * *

## LoRA Fine-Tuning [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#lora-fine-tuning "Link to this heading")

LoRA trains a small number of additional parameters (typically < 1% of the model) while keeping the base model frozen. It is the recommended starting point for most fine-tuning tasks.

### Training [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#training "Link to this heading")

**Configuration**

Create a YAML config file. Here is an example for VoxCPM 2:

```
pretrained_path: /path/to/VoxCPM2/
train_manifest: /path/to/train.jsonl
val_manifest: /path/to/val.jsonl   # optional, leave empty to skip validation

sample_rate: 16000        # AudioVAE encoder input rate (NOT the 48kHz output rate)
out_sample_rate: 48000    # AudioVAE decoder output rate; only used at inference, not during training
batch_size: 16
grad_accum_steps: 1
num_workers: 2
num_iters: 1000
log_interval: 10
valid_interval: 500
save_interval: 500

learning_rate: 0.0001
weight_decay: 0.01
warmup_steps: 100
max_steps: 1000
max_batch_tokens: 8192

save_path: /path/to/checkpoints/lora
tensorboard: /path/to/logs/lora

lambdas:
  loss/diff: 1.0
  loss/stop: 1.0

lora:
  enable_lm: true
  enable_dit: true
  enable_proj: false
  r: 32
  alpha: 32
  dropout: 0.0
```

Copy to clipboard

Tip

For VoxCPM 1.5, change `sample_rate` to `44100` and `pretrained_path` to your VoxCPM 1.5 checkpoint. The `sample_rate` must always match the AudioVAE encoder input rate in `config.json` — **not** the output rate. The training script auto-detects the model architecture from `config.json`.

**LoRA parameters explained**

| Parameter | Description | Recommended |
| --- | --- | --- |
| `enable_lm` | Apply LoRA to the language model (base LM + residual LM) | `true` |
| `enable_dit` | Apply LoRA to the diffusion transformer | `true` (essential for voice quality) |
| `enable_proj` | Apply LoRA to projection layers between LM and DiT | `false` for most cases |
| `r` | LoRA rank — higher means more capacity | 32 for speaker cloning, 64 for style/language adaptation |
| `alpha` | Scaling factor (`scaling = alpha / r`) | Usually `r` or `2*r`. Adjust to control LoRA influence strength. |
| `dropout` | Dropout on LoRA layers | `0.0` unless overfitting |

**Launch**

```
# Single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/your_lora_config.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/your_lora_config.yaml
```

Copy to clipboard

**LoRA WebUI**

VoxCPM also provides a Gradio UI that wraps LoRA training and inference in one place:

```
python lora_ft_webui.py
```

Copy to clipboard

### Monitoring [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#monitoring "Link to this heading")

Training logs to TensorBoard. Start the viewer with:

```
tensorboard --logdir /path/to/logs/lora
```

Copy to clipboard

**What to watch**

| Metric | What it tells you |
| --- | --- |
| `loss/diff` | Diffusion loss — should steadily decrease, then flatten |
| `loss/stop` | Stop prediction loss — should stabilize early and stay low |
| `grad_norm` | Gradient magnitude — spikes may indicate bad samples or too high a learning rate |
| `lr` | Learning rate curve — cosine decay with warmup, useful to verify your schedule |

If a validation manifest is provided, the script also logs `val/loss` and generates sample audio + mel spectrograms in TensorBoard at each `valid_interval`.

**When to stop**

- **Use epochs as a rough guide.** For single-speaker cloning, 1–3 epochs are usually sufficient. Going beyond that often hurts rather than helps — overfitting in TTS fine-tuning can emerge very early.

- `loss/diff` plateaus and no longer decreases meaningfully.

- Generated audio in TensorBoard sounds good on your target voice/style.

- If the model starts ignoring input text (generating the same audio regardless of text), you have overfit — roll back to an earlier checkpoint.


Tip

Validation loss does not always correlate perfectly with perceptual quality. Save multiple checkpoints around the convergence zone and evaluate them with actual inference to pick the best one.

**Checkpoint structure**

```
checkpoints/lora/
├── step_0000500/
│   ├── lora_weights.safetensors
│   ├── lora_config.json
│   ├── optimizer.pth
│   ├── scheduler.pth
│   └── training_state.json
├── step_0001000/
│   └── ...
└── latest -> step_0001000/
```

Copy to clipboard

Training automatically resumes from `latest/` if it exists. The signal handler also saves a checkpoint on `SIGTERM` / `SIGINT` so you don’t lose progress on interruption.

### Inference [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#inference "Link to this heading")

**CLI**

```
python scripts/test_voxcpm_lora_infer.py \
    --lora_ckpt /path/to/checkpoints/lora/step_0002000 \
    --text "Hello from the fine-tuned model." \
    --output output.wav
```

Copy to clipboard

**Python API**

```
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained(
    "openbmb/VoxCPM2",
    lora_weights_path="/path/to/checkpoints/lora/latest",
)

wav = model.generate(text="Hello from the fine-tuned model.")
```

Copy to clipboard

**Hot-swapping LoRA at runtime**

You can load, unload, and switch LoRA weights without restarting the model:

```
# Load a LoRA
model.load_lora("/path/to/lora_a")

# Disable LoRA temporarily (base model only)
model.set_lora_enabled(False)

# Re-enable
model.set_lora_enabled(True)

# Switch to a different LoRA
model.unload_lora()
model.load_lora("/path/to/lora_b")
```

Copy to clipboard

All hot-swap operations are compatible with `torch.compile`.

* * *

## Full Fine-Tuning [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#full-fine-tuning "Link to this heading")

Full fine-tuning updates all model parameters. Use it when LoRA does not provide enough capacity — typically for new languages or large-scale customization with 500+ clips.

### Training [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#id1 "Link to this heading")

**Configuration**

```
pretrained_path: /path/to/VoxCPM2/
train_manifest: /path/to/train.jsonl
val_manifest: /path/to/val.jsonl

sample_rate: 16000        # AudioVAE encoder input rate (NOT the 48kHz output rate)
out_sample_rate: 48000    # AudioVAE decoder output rate; only used at inference, not during training
batch_size: 16
grad_accum_steps: 1
num_workers: 2
num_iters: 1000
log_interval: 10
valid_interval: 500
save_interval: 500

learning_rate: 0.00001    # 10x smaller than LoRA
weight_decay: 0.01
warmup_steps: 100
max_steps: 1000
max_batch_tokens: 8192

save_path: /path/to/checkpoints/full
tensorboard: /path/to/logs/full

lambdas:
  loss/diff: 1.0
  loss/stop: 1.0
```

Copy to clipboard

Note the `lora` key is absent — this tells the script to do full fine-tuning.

**Key differences from LoRA**

- `learning_rate` should be ~10x smaller (`1e-5` vs `1e-4`) to avoid catastrophic forgetting.

- VRAM usage is significantly higher because all parameters require gradients.

- Checkpoints are larger (full model weights vs. LoRA delta only).


**Launch**

```
# Single GPU
python scripts/train_voxcpm_finetune.py --config_path conf/your_full_config.yaml

# Multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_voxcpm_finetune.py --config_path conf/your_full_config.yaml
```

Copy to clipboard

### Monitoring [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#id2 "Link to this heading")

Same TensorBoard metrics as LoRA (`loss/diff`, `loss/stop`, `grad_norm`, `lr`, validation audio).

Full fine-tuning is more prone to overfitting than LoRA. In practice, full fine-tuning often reaches its optimum within 1–2 epochs — continuing beyond that can degrade quality. Pay extra attention to:

- **Validation loss diverging from training loss** — a sign of overfitting. Stop and use the last checkpoint before divergence.

- **Text being ignored** — the most common overfitting symptom. Keep `training_cfg_rate=0.1` (do not set it to 0) and `weight_decay=0.01`. Monitor checkpoints at each `save_interval`.

- **Smaller datasets overfit faster.** With fewer training samples, the optimal checkpoint may appear within a few hundred steps.

- **New language fine-tuning:** Mix in some Chinese/English data (e.g. 10–20%) to reduce forgetting of the original capabilities.

- **More data does not always mean better results.** Beyond a certain point, adding more data yields diminishing returns; focus on data quality and diversity instead.


**Checkpoint structure**

```
checkpoints/full/
├── step_0000500/
│   ├── model.safetensors
│   ├── config.json
│   ├── audiovae.pth
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── optimizer.pth
│   ├── scheduler.pth
│   └── training_state.json
└── latest -> step_0000500/
```

Copy to clipboard

Each checkpoint is a complete model directory that can be loaded directly.

### Inference [¶](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html\#id3 "Link to this heading")

**CLI**

```
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir /path/to/checkpoints/full/step_0002000 \
    --text "Hello from the fine-tuned model." \
    --output output.wav

# With voice cloning
python scripts/test_voxcpm_ft_infer.py \
    --ckpt_dir /path/to/checkpoints/full/latest \
    --text "Cloned voice with full fine-tuning." \
    --prompt_audio reference.wav \
    --prompt_text "Exact transcript of reference.wav" \
    --output cloned.wav
```

Copy to clipboard

**Python API**

The checkpoint directory is a complete model — load it directly:

```
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("/path/to/checkpoints/full/latest")
wav = model.generate(text="Hello from the fine-tuned model.")
```

Copy to clipboard

* * *

For common training issues (OOM, runaway generation, poor LoRA performance, checkpoint errors), see [Fine-Tuning FAQ](https://voxcpm.readthedocs.io/en/latest/finetuning/faq.html).

Languages**[en](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)**[zh-cn](https://voxcpm.readthedocs.io/zh-cn/latest/finetuning/finetune.html)Versions**[latest](https://voxcpm.readthedocs.io/en/latest/finetuning/finetune.html)**On Read the Docs[Project Home](https://app.readthedocs.org/projects/voxcpm/?utm_source=voxcpm&utm_content=flyout)[Builds](https://app.readthedocs.org/projects/voxcpm/builds/?utm_source=voxcpm&utm_content=flyout)Search

* * *

[Addons documentation](https://docs.readthedocs.io/page/addons.html?utm_source=voxcpm&utm_content=flyout) ― Hosted by
[Read the Docs](https://about.readthedocs.com/?utm_source=voxcpm&utm_content=flyout)