# Lightning AI Studio — VSCode + Claude Code on a GPU (no Colab)

Goal: get a cloud GPU you reach from **your own desktop VSCode**, with **Claude Code running on it**,
so we do data-prep + VoxCPM training there while you stay in VSCode. GPU is toggled on only when needed
(it's a free CPU Studio otherwise).

> Why this setup: your laptop has no NVIDIA GPU, so training (~12–20 GB VRAM) can't run locally. Lightning
> gives a free 24/7 Studio + free monthly GPU-hours, built-in VSCode, **and** SSH so your desktop VSCode +
> Claude Code attach to it. See `../colab/README.md` for the pipeline these run.

---

## Step 0 — Push the latest code to GitHub (do this on your laptop first)
The Studio clones from GitHub, so it must be current. From the repo root:
```bash
git add .gitignore tts-dataprep-pipeline.md colab/ cloud/
git commit -m "Colab/cloud data-prep notebook + Lightning setup"
git push origin main
```
(Claude can do this for you — just say so.) The 1 lecture's audio is already in git, so it clones too.

## Step 1 — Create the Lightning account + Studio
1. Sign up at **https://lightning.ai** (free).
2. **Verify your phone number** — this unlocks the free monthly GPU-hours (~15–22 h on T4/L4).
3. Click **New Studio** → it boots a free 4-CPU Linux environment (VSCode is built in, in the browser).
   You can work in the browser VSCode immediately, but we'll attach your *desktop* VSCode next.

## Step 2 — Attach your desktop VSCode over SSH
1. In VSCode (desktop), install the **"Remote - SSH"** extension (Microsoft).
2. Add your SSH public key to Lightning: Studio → the **SSH / "Connect"** option → follow its prompt to
   paste your `~/.ssh/id_*.pub` (generate one with `ssh-keygen -t ed25519` if you don't have it). Lightning
   then shows a host line like `ssh s_<id>@ssh.lightning.ai`.
3. Add that host to `~/.ssh/config` (Lightning gives you the exact block to paste).
4. In desktop VSCode: `Ctrl/Cmd+Shift+P` → **Remote-SSH: Connect to Host** → pick the Lightning host.
   A new VSCode window opens **running on the Studio** — same VSCode, GPU underneath.
   - Docs: https://lightning.ai/docs/overview/ai-studio/  (search "SSH" / "connect local IDE")

## Step 3 — Install Claude Code on the Studio
In the Studio's terminal (the integrated terminal in that remote VSCode window):
```bash
curl -fsSL https://claude.ai/install.sh | bash      # native installer, no node needed
claude                                                # run it, follow the login prompt to authenticate
```
Now Claude Code is running **on the GPU box** — same assistant, driving the remote environment.

## Step 4 — Get the project onto the Studio
```bash
git clone https://github.com/Mukul1312/voice-cloning.git
cd voice-cloning
```
This brings the code + the 1 lecture's audio. (For the *other* lectures later, upload their audio straight
into the Studio — drag-drop in the file browser or `scp` — rather than bloating git with big media.)

## Step 5 — Hand off to Claude on the Studio
In that remote terminal, start Claude Code and point it at the plan:
```
claude
> Read cloud/HANDOFF.md, then continue the GGS voice-cloning project from there.
```
`HANDOFF.md` tells the Studio's Claude exactly where we are and what to do next (env setup, turn on the GPU,
run data-prep, then the VoxCPM2 LoRA train). You stay in VSCode the whole time.

---

## Turning the GPU on/off (cost control)
- Studios start on **CPU (free)**. Switch to a GPU (T4/L4/…) from the Studio's compute selector **only**
  for the heavy steps (alignment/diarization batch, training). Switch back to CPU when done.
- Free hours are limited (~15–22/mo). A LoRA run is a few hours; on a paid **L4 (24 GB, ~$0.70/hr)** a full
  run is ≈ $2–5. T4 (16 GB, free) fits **VoxCPM 1.5** LoRA or a shrunk-batch VoxCPM2 (see `finetune-plan.md`).
- Your **/teamspace/studios/this_studio** storage (100 GB) persists across GPU on/off and restarts.
