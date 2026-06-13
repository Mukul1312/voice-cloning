# RunPod — VSCode + Claude Code on a GPU (India-friendly; $10 funded)

Same end state as the Lightning plan (desktop VSCode → SSH into a GPU box → Claude Code on it), but on
RunPod, which takes international cards/crypto (no phone-SMS wall). Two layers:

- **Laptop Claude + RunPod MCP** = the *infrastructure* layer: create/start/STOP the pod, check balance, get
  SSH details — driven from Claude Code by natural language, so you don't hand-click the UI.
- **Pod Claude (via VSCode Remote-SSH)** = the *work* layer: runs the data-prep + training (reads `HANDOFF.md`).

> ⚠️ RunPod bills **per second while a pod runs** (no free idle mode). Stop the pod when not in use; persist
> outputs to the attached volume or `git push`. The MCP can stop it for you ("stop my pod").

---

## Step 1 — Get a RunPod API key
RunPod console → **Settings → API Keys → +Create** (read/write). Copy it. **Do NOT paste it into the chat** —
it goes only into the local command below.

## Step 2 — Connect RunPod's MCP servers to Claude Code (run in YOUR terminal)
Needs Node.js (for `npx`). Run these on your laptop:
```bash
# Infra control (needs your key — keep it local, never in chat):
claude mcp add runpod --scope user -e RUNPOD_API_KEY=YOUR_KEY_HERE -- npx -y @runpod/mcp-server@latest

# Docs search (no auth):
claude mcp add runpod-docs --scope user --transport http https://docs.runpod.io/mcp
```
Then **restart Claude Code** so the tools load. Run `/mcp` to confirm `runpod` + `runpod-docs` are connected.

## Step 3 — Have Claude create the pod
Once the MCP is live, ask Claude (e.g.):
> Check my RunPod balance and available GPUs, then create a Pod: RTX 4090 (24 GB), a recent
> runpod/pytorch CUDA-12 image, 1 GPU, Community cloud, a 30 GB volume, SSH enabled. Give me the SSH details.

Why these: **RTX 4090 = 24 GB** fits VoxCPM2's ~20 GB; **Community** cloud is cheapest; the **volume** keeps
clips/outputs across stop/start; **SSH** is what VSCode Remote-SSH attaches to. (Our env setup pins torch 2.8
itself, so any recent CUDA-12 PyTorch base image is fine.)

## Step 4 — Attach your desktop VSCode
1. VSCode → install **"Remote - SSH"** extension.
2. Add the pod's SSH details (host/port/key from Step 3) to `~/.ssh/config`.
3. `Ctrl/Cmd+Shift+P` → **Remote-SSH: Connect to Host** → pick the pod. A VSCode window opens *on the GPU pod*.
   - Reference: https://www.runpod.io/articles/guides/seamless-cloud-ide-using-vs-code-remote

## Step 5 — On the pod: Claude Code + repo
In the pod's VSCode terminal:
```bash
curl -fsSL https://claude.ai/install.sh | bash        # Claude Code (Linux)
claude                                                  # log in
git clone https://github.com/Mukul1312/voice-cloning.git
cd voice-cloning
# then: tell Claude  ->  "Read cloud/HANDOFF.md and continue."
```

## Cost hygiene (make the $10 last)
- A 24 GB 4090 ≈ $0.34–0.69/hr. Data-prep for all lectures ≈ ~1 GPU-hr; a LoRA run ≈ a few hrs. $10 is plenty.
- **Stop the pod whenever you step away** (ask laptop-Claude: "stop my RunPod pod"). A *stopped* pod keeps its
  volume but doesn't bill GPU time.
- Push results to GitHub or keep them on the 30 GB volume so a stop never loses work.
