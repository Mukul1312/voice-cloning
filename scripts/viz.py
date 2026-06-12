"""
viz.py - render a 2D audio representation per clip so Claude can SEE what it can't hear.

For given clip indices it draws, per clip:
  - top:    mel-spectrogram (time x mel-freq x energy)   -> shows clean cuts / silence at edges
  - bottom: pitch (F0) contour                            -> speech = wiggly, chant = sustained/stepped

RUN (inside the venv):
  ./.venv/Scripts/python.exe scripts/viz.py i-and-mine-and-namabhasa-stage 1 39 80
"""
import json, sys
from pathlib import Path
import numpy as np
import librosa, librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

slug = sys.argv[1] if len(sys.argv) > 1 else "i-and-mine-and-namabhasa-stage"
idxs = [int(x) for x in sys.argv[2:]] or [1, 39, 80]

d = Path(__file__).resolve().parent.parent / "data" / "lectures" / slug
clips, out = d / "clips", d / "viz"
out.mkdir(exist_ok=True)
rows = [json.loads(l) for l in open(d / "train.jsonl", encoding="utf-8")]

for i in idxs:
    wav = clips / f"{slug}_{i:04d}.wav"
    if not wav.exists():
        print("missing", wav); continue
    y, sr = librosa.load(str(wav), sr=16000)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
    tf0 = librosa.times_like(f0, sr=sr)

    fig, ax = plt.subplots(2, 1, figsize=(13, 5), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    librosa.display.specshow(mel, sr=sr, x_axis="time", y_axis="mel", ax=ax[0], cmap="magma")
    txt = rows[i-1]["text"] if i-1 < len(rows) else ""
    ax[0].set_title(f"clip {i}  ({rows[i-1]['duration']}s):  {txt[:95]}", fontsize=10)
    ax[0].set_ylabel("mel freq")
    ax[1].plot(tf0, f0, color="cyan")
    ax[1].set_ylim(0, 400); ax[1].set_ylabel("pitch F0 (Hz)"); ax[1].set_xlabel("time (s)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / f"clip_{i:04d}.png"
    fig.savefig(str(p), dpi=85); plt.close(fig)
    print("wrote", p)
