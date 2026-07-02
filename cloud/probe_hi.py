"""
probe_hi.py — DE-RISK PROBE (~1 h): is base VoxCPM2's Hindi good enough to build a GGS Hindi clone on?

Run this BEFORE any Hindi data-prep. VoxCPM2 officially supports Hindi (30-language list; accepts
Devanagari directly, no language tag) BUT Hindi is one of its two acknowledged WEAKEST languages (tech
report: "The main weaknesses appear in Arabic and Hindi ... relatively limited data volume"; Minimax
WER 19.7 vs English ~2.3), and GitHub issue #288 reports Hindi cloning misbehaving. So we audition the
BASE model (no LoRA — we have no Hindi LoRA yet) on a hand-cut Hindi reference clip of HIS voice, across
two axes that are genuinely unknown for our case:

  axis 1  SCRIPT : Devanagari (native) vs romanized/Hinglish   -> the tokenizer / G2P question
  axis 2  MODE   : hifi  (prompt_text = the ref's Hindi transcript; "Ultimate cloning"; best identity)
                   plain (no prompt_text; voice-clone only)     -> the issue-#288 artifact question
  x SEEDS        : seed variance is real (the English blind-eval lesson) -> 3 seeds each
  4 sentences x 2 scripts x 2 modes x 3 seeds = 48 takes.

GATE (ear-check the outputs): if base VoxCPM2 speaks clean, in-his-voice Hindi -> proceed with the
VoxCPM2 LoRA plan (same recipe as English). If it's unintelligible / anglicized / artifact-laden ->
switch the base to IndicF5 (AI4Bharat, native Hindi, MIT) or svara-tts-v1, KEEPING all of our data-prep /
eval / funnel methodology. Either way the probe is cheap and decisive.

SETUP (pod — same convention as infer_refsweep.py):
  1. deploy a pod on the voxcpm-vol network volume (web console), SSH in, run pod_infer.sh (env + base)
  2. hand-cut ONE clean ~15 s Hindi clip of his voice -> /workspace/ref_hi.wav
  3. pass its exact Hindi transcript (verbatim Devanagari, what he says in the clip) via --ref-text
  4. python cloud/probe_hi.py --ref /workspace/ref_hi.wav --ref-text "<devanagari transcript>"

Outputs: /workspace/out_probe_hi/{sentid}__{script}__{mode}__s{seed}.wav  (self-describing; resumable).
Pull them local and ear-check (or reuse a build_*_gallery.py pattern for an A/B page).

!!! The Hindi TEST SENTENCES below are AUTHOR-PROVIDED best-effort — VERIFY / EDIT the Devanagari spelling
    and the romanization before trusting the read. You are the Hindi speaker; I am not. !!!
"""
import argparse
import gc
import os
import sys

import soundfile as sf
from voxcpm.core import VoxCPM

BASE = "/workspace/VoxCPM2"
OUT = "/workspace/out_probe_hi"

# (id, devanagari, romanized) — his devotional register + number/date reads (the diary will need both)
SENTS = [
    ("shelter", "कृष्ण ही हमारे एकमात्र आश्रय हैं।",
                "Krishna hi hamaare ekmaatra aashray hain."),
    ("guru",    "गुरु की कृपा के बिना कोई इस भवसागर को पार नहीं कर सकता।",
                "Guru ki kripa ke bina koi is bhavsaagar ko paar nahin kar sakta."),
    ("date",    "आज पाँच मई उन्नीस सौ पचहत्तर है।",   # numbers SPELLED OUT (English clone read digits badly)
                "Aaj paanch mai unnees sau pachhattar hai."),
    ("digits",  "आज 5 मई 1975 है।",                   # RAW digits — tests whether Hindi will need spell-out too
                "Aaj 5 May 1975 hai."),
]
SEEDS = [42, 123, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="hand-cut ~15 s Hindi reference clip of his voice")
    ap.add_argument("--ref-text", required=True, help="verbatim Devanagari transcript of the --ref clip")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--denoise", action="store_true",
                    help="run VoxCPM's built-in denoiser on the (uncleaned) reference clip")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("loading base VoxCPM2 (no LoRA) ...", file=sys.stderr)
    m = VoxCPM.from_pretrained(hf_model_id=BASE, load_denoiser=True, optimize=False)
    sr = m.tts_model.sample_rate

    jobs = [(sid, script, text, mode, s)
            for (sid, dev, rom) in SENTS
            for (script, text) in (("dev", dev), ("rom", rom))
            for mode in ("hifi", "plain")
            for s in a.seeds]
    todo = [j for j in jobs
            if not os.path.exists(f"{OUT}/{j[0]}__{j[1]}__{j[3]}__s{j[4]}.wav")]
    print(f"{len(todo)}/{len(jobs)} takes to generate", file=sys.stderr)

    for (sid, script, text, mode, s) in todo:
        kw = dict(text=text, prompt_wav_path=a.ref, denoise=a.denoise, seed=s)
        if mode == "hifi":                       # plain mode = omit prompt_text entirely
            kw["prompt_text"] = a.ref_text
        w = m.generate(**kw)
        out = f"{OUT}/{sid}__{script}__{mode}__s{s}.wav"
        sf.write(out, w, sr)
        print(f"wrote {os.path.basename(out)}", file=sys.stderr)

    del m
    gc.collect()
    print("DONE — pull /workspace/out_probe_hi/ and ear-check. "
          "Hypotheses: Devanagari >= romanized; hifi carries his identity best; "
          "listen for the issue-#288 spurious leading word.")


if __name__ == "__main__":
    main()
