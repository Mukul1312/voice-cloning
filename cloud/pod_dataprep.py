"""
pod_dataprep.py — GGS TTS data-prep on a (RunPod) GPU box. Pod-native port of
colab/ggs_dataprep_colab.ipynb: same verified pipeline, but NO Colab deps (local paths +
HF_TOKEN env instead of Drive mounts / userdata secrets).

Pipeline per lecture (data/lectures/<slug>/{audio.mp3,transcript.txt}):
  A diarize (pyannote) -> speaker turns
  B verify which cluster is GGS (ECAPA cosine vs a clean GGS reference clip) -> GGS-only intervals
  C forced-align the KNOWN transliterated transcript (ctc-forced-aligner / MMS) -> per-word timestamps
  D intersect words with GGS-only intervals (drops questioner words by timeline geometry)
  E group GGS-only words into sentence clips, cut 16k wavs (both-end silence trim) -> train.jsonl
  F ASR round-trip CER/WER QA (faster-whisper) -> train.clean.jsonl + qa_asr_report.tsv

We KEEP verses (GGS speaks them). transliterate() + clip recipe reused from scripts/.

USAGE
  export HF_TOKEN=hf_xxx                       # accept the pyannote community-1 gated repos first
  python cloud/pod_dataprep.py --slug i-and-mine-and-namabhasa-stage --ref data/refs/ggs_ref_01.wav
  python cloud/pod_dataprep.py --scale --ref data/refs/ggs_ref_01.wav   # all lectures + group split
"""
import argparse, json, os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from classify_transliterate import transliterate          # IAST -> lossy ASCII (verses kept)

LECT = ROOT / "data" / "lectures"

# ---- clip recipe (verbatim from scripts/align_cut.py) -------------------------------
HARD_MIN, HARD_MAX = 4.0, 30.0
LEAD_PAD, TAIL_PAD = 0.15, 0.15
SIL_DB = -40

# ---- Sanskrit/Vaishnava loanwords masked out for cer_core (English-backbone CER) ----
SANSKRIT_TOKENS = {
    "krishna","krsna","rama","hare","govinda","gopala","bhagavan","bhagavad","bhagavatam","shrimad",
    "srimad","gita","vedas","veda","vedanta","upanishad","bhakti","bhakta","jiva","jivatma","atma",
    "atman","paramatma","brahman","brahma","maya","moha","karma","jnana","yoga","yogi","guru","acarya",
    "acharya","prabhupada","chaitanya","caitanya","mahaprabhu","vaishnava","vaisnava","namabhasa","nama",
    "japa","kirtan","kirtana","mantra","prasada","prasadam","samadhi","sadhana","sadhu","shastra","sastra",
    "smriti","sruti","dharma","adharma","sankirtana","harinama","vrindavan","vrndavana","mathura","dvaraka",
    "goloka","vaikuntha","purusa","purusha","prakriti","prakrti","gunas","guna","sattva","rajas","tamas",
    "om","aum","namo","namah"}


# ===================================================================================
# shared helpers
# ===================================================================================
def to_wav_16k(src: Path, dst: Path) -> Path:
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-ac", "1", "-ar", "16000", str(dst)], check=True)
    return dst


def cut(src: Path, start: float, end: float, dst: Path):
    sr = "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-40dB"
    trim = f"{sr},areverse,{sr},areverse"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                    "-af", trim, "-ac", "1", "-ar", "16000", str(dst)], check=True)


def wav_duration(path: Path) -> float:
    return float(librosa.get_duration(path=str(path)))


def build_clips(words, min_sec, max_sec):
    """Group words into clips ending at a sentence boundary (.?!), ~min..max sec. (verbatim)"""
    clips, cur = [], []
    def flush():
        if cur:
            s, e = cur[0], cur[-1]
            if HARD_MIN <= words[e]["end"] - words[s]["start"] <= HARD_MAX:
                clips.append((s, e))
        cur.clear()
    for idx, w in enumerate(words):
        cur.append(idx)
        dur = words[cur[-1]]["end"] - words[cur[0]]["start"]
        if dur >= max_sec or (w["text"].rstrip().endswith((".", "?", "!")) and dur >= min_sec):
            flush()
    flush()
    return clips


def ggs_only_text(raw: str, speaker_re, ggs_re) -> str:
    """Keep only GGS turns' text when the transcript has speaker labels; else return raw."""
    if speaker_re is None and ggs_re is None:
        return raw
    header_re = re.compile(r"^\s*([A-Za-z][A-Za-z .]{1,30}):\s*(.*)$")
    keep, cur_is_ggs = [], True
    for ln in raw.splitlines():
        m = header_re.match(ln)
        if m:
            if ggs_re and ggs_re.match(ln):       cur_is_ggs = True
            elif speaker_re and speaker_re.match(ln): cur_is_ggs = False
            else:                                  cur_is_ggs = False
            if cur_is_ggs: keep.append(m.group(2))
        elif cur_is_ggs:
            keep.append(ln)
    return "\n".join(keep)


# ===================================================================================
# Stage A — diarization (pyannote)
# ===================================================================================
def load_diar_pipeline(hf_token, device):
    import torch
    import pyannote.audio as pa
    from pyannote.audio import Pipeline
    # pyannote 4.x (needs torch 2.8 -> host CUDA>=12.6): community-1 + token=
    # pyannote 3.x (runs on torch 2.4 -> host CUDA 12.4):  speaker-diarization-3.1 + use_auth_token=
    if int(pa.__version__.split(".")[0]) >= 4:
        model = "pyannote/speaker-diarization-community-1"
        pipe = Pipeline.from_pretrained(model, token=hf_token)
    else:
        model = "pyannote/speaker-diarization-3.1"
        pipe = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    if pipe is None:
        raise RuntimeError(
            f"pyannote Pipeline.from_pretrained returned None for {model} — accept the terms for EVERY gated "
            f"repo it lists (for 3.1 that's speaker-diarization-3.1 + segmentation-3.0) and use a valid READ token.")
    pipe.to(torch.device(device))
    print(f"  [diarize] {model} (pyannote {pa.__version__}) on {device}")
    return pipe


def load_audio_in_memory(wav_path, target_sr=16000):
    import torch, torchaudio
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr); sr = target_sr
    return {"waveform": wav, "sample_rate": sr}


def diarize(pipe, wav_path, min_speakers, max_speakers):
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    audio = load_audio_in_memory(wav_path)
    with ProgressHook() as hook:
        out = pipe(audio, hook=hook, min_speakers=min_speakers, max_speakers=max_speakers)
    if hasattr(out, "exclusive_speaker_diarization"):
        ann = out.exclusive_speaker_diarization          # pyannote 4.0 community-1 (non-overlapping turns)
    elif hasattr(out, "speaker_diarization"):
        ann = out.speaker_diarization                    # community-1 result without the exclusive view
    else:
        ann = out                                        # pyannote 3.1: pipeline returns an Annotation directly
    turns, secs = [], {}
    for seg, _trk, spk in ann.itertracks(yield_label=True):
        turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": spk})
        secs[spk] = secs.get(spk, 0.0) + float(seg.duration)
    turns.sort(key=lambda t: t["start"])
    return turns, secs


# ===================================================================================
# Stage B — ECAPA speaker-verify -> GGS-only intervals
# ===================================================================================
def make_embedder(device):
    import torch
    import torch.nn.functional as F
    from speechbrain.inference.speaker import EncoderClassifier
    import torchaudio
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec-ecapa-voxceleb", run_opts={"device": device})

    def load_wav(path, start=None, end=None, sr_t=16000):
        if start is not None and end is not None:
            sr = torchaudio.info(path).sample_rate
            wav, sr = torchaudio.load(path, frame_offset=int(start*sr), num_frames=int((end-start)*sr))
        else:
            wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if sr != sr_t: wav = torchaudio.transforms.Resample(sr, sr_t)(wav)
        return wav

    @torch.no_grad()
    def embed(path, start=None, end=None):
        emb = clf.encode_batch(load_wav(path, start, end).to(device)).squeeze(0).squeeze(0)
        return F.normalize(emb, dim=0)

    def cosine(a, b): return float(torch.dot(a, b))
    return embed, cosine, (lambda paths: F.normalize(torch.stack([embed(p) for p in paths]).mean(0), dim=0))


def build_ggs_intervals(turns, ggs_speaker, merge_gap=0.25):
    ggs = sorted((t["start"], t["end"]) for t in turns if t["speaker"] == ggs_speaker)
    other = sorted((t["start"], t["end"]) for t in turns if t["speaker"] != ggs_speaker)
    def gap_has_other(g0, g1): return any(o0 < g1 and o1 > g0 for o0, o1 in other)
    out = []
    for s, e in ggs:
        if out and (s - out[-1][1]) <= merge_gap and not gap_has_other(out[-1][1], s):
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def verify_ggs(turns, secs, wav16, ref_clips, embed, cosine, centroid, cfg):
    ggs_ref = centroid(ref_clips)
    sums, counts = defaultdict(float), defaultdict(int)
    for t in turns:
        if (t["end"] - t["start"]) < cfg["min_turn_sec"]:
            continue
        sims = cosine(embed(str(wav16), t["start"], t["end"]), ggs_ref)
        sums[t["speaker"]] += sims; counts[t["speaker"]] += 1
    speaker_cos = {s: sums[s]/counts[s] for s in counts}
    if not speaker_cos:
        raise RuntimeError("no cluster long enough to verify GGS — check reference clip / diarization")
    gspk = max(speaker_cos, key=speaker_cos.get)
    ranked = sorted(speaker_cos.values(), reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    flags = []
    if speaker_cos[gspk] < cfg["cosine_thr"]:                 flags.append(f"low_cosine({speaker_cos[gspk]:.3f})")
    if len(ranked) > 1 and margin < cfg["cosine_margin"]:     flags.append(f"low_margin({margin:.3f})")
    if max(secs, key=secs.get) != gspk:                       flags.append("cosine!=talktime")
    print(f"  [verify] GGS cluster={gspk} cos={speaker_cos[gspk]:+.3f} margin={margin:+.3f} "
          f"talk={secs.get(gspk,0)/60:.1f}min")
    for s, c in sorted(speaker_cos.items(), key=lambda kv: -kv[1]):
        print(f"     {s}: cos={c:+.3f} talk={secs.get(s,0)/60:5.1f}min")
    return build_ggs_intervals(turns, gspk), flags


# ===================================================================================
# Stage C — forced alignment (ctc-forced-aligner / MMS)
# ===================================================================================
def make_aligner(device):
    import torch, torchaudio
    from ctc_forced_aligner import (load_alignment_model, generate_emissions, preprocess_text,
                                     get_alignments, get_spans, postprocess_results)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model, tokenizer = load_alignment_model(device, dtype=dtype)

    def _load_16k_mono(path):
        # load with torchaudio (our input is already 16k mono) -> 1D waveform on model device/dtype.
        # avoids ctc-forced-aligner's load_audio(), which requires torchcodec (incompatible w/ torch 2.4).
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return wav.squeeze(0).to(model.device, model.dtype)

    def align(audio_path, transcript, language="eng", batch_size=8):
        wav = _load_16k_mono(audio_path)
        emissions, stride = generate_emissions(model, wav, batch_size=batch_size)
        tok_starred, txt_starred = preprocess_text(transcript, romanize=True, language=language, split_size="word")
        segments, scores, blank = get_alignments(emissions, tok_starred, tokenizer)
        spans = get_spans(tok_starred, segments, blank)
        raw = postprocess_results(txt_starred, spans, stride, scores)
        words = []
        for w in raw:
            if w["text"] in ("<star>", ""):
                continue
            words.append({"text": w["text"], "start": float(w["start"]), "end": float(w["end"]),
                          "score": float(w.get("score", 0.0))})
        return words
    return align


# ===================================================================================
# Stage D — intersect words with GGS-only intervals
# ===================================================================================
def _overlap(a0, a1, b0, b1): return max(0.0, min(a1, b1) - max(a0, b0))

def word_is_ggs(w, intervals, overlap_frac=0.9):
    wd = max(1e-6, w["end"] - w["start"])
    ov = sum(_overlap(w["start"], w["end"], s, e) for s, e in intervals)
    return (ov / wd) >= overlap_frac


# ===================================================================================
# Stage E — clips
# ===================================================================================
def ggs_interval_end_after(t, intervals):
    for s, e in intervals:
        if s - 1e-3 <= t <= e + 1e-3:
            return e
    return t


def emit_clips(words, intervals, wav16, clips_dir, slug, min_sec, max_sec):
    clips_dir.mkdir(parents=True, exist_ok=True)
    for old in clips_dir.glob("*.wav"): old.unlink()
    # contiguous GGS-only runs (questioner words break a run)
    runs, cur = [], []
    for w in words:
        if w["is_ggs"]: cur.append(w)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    rows, idx = [], 1
    for run in runs:
        for (s, e) in build_clips(run, min_sec, max_sec):
            prev_end = run[s-1]["end"] if s > 0 else max(0.0, run[s]["start"] - 1.0)
            nxt = run[e+1]["start"] if e+1 < len(run) else run[e]["end"] + 1
            start = max(run[s]["start"] - LEAD_PAD, prev_end, 0.0)
            iv_end = ggs_interval_end_after(run[e]["end"], intervals)
            end = min(run[e]["end"] + min(TAIL_PAD, max(0.0, nxt - run[e]["end"])), iv_end)
            txt = " ".join(run[k]["text"] for k in range(s, e + 1))
            scs = [float(run[k].get("score", 0.0)) for k in range(s, e + 1)]   # per-word alignment confidence
            dst = clips_dir / f"{slug}_{idx:04d}.wav"
            cut(wav16, start, end, dst)
            rows.append({"audio": f"clips/{dst.name}", "text": txt, "duration": round(wav_duration(dst), 2),
                         "align_mean": round(sum(scs)/len(scs), 4), "align_min": round(min(scs), 4)})
            idx += 1
    return rows


# ===================================================================================
# Stage F — ASR round-trip CER/WER QA
# ===================================================================================
def silence_edges(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    voiced = np.where(db > SIL_DB)[0]
    if len(voiced) == 0:
        return None
    t = librosa.frames_to_time(voiced, sr=sr, hop_length=256)
    return float(t[0]), float(len(y) / sr - t[-1])


def load_english():
    """dwyl English wordlist (already in the repo) for CER masking — any non-dictionary word (Sanskrit) is
    excluded from cer_core. False-negatives don't matter here: we only want the English-backbone CER."""
    p = ROOT / "data" / "english_words.txt"
    return set(p.read_text(encoding="utf-8").split()) if p.exists() else set()


def make_scorer(english):
    """cer_core = CER over ENGLISH-DICTIONARY words only (Sanskrit terms Whisper can't spell don't count);
    cer_raw = over all words; sanskrit_load = the difference. CER is ADVISORY only (Whisper mis-reads Sanskrit
    AND can hallucinate continuations) — the real quality gate is the aligner's own per-clip confidence."""
    import jiwer
    cer_tx = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                            jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfChars()])
    wer_tx = jiwer.wer_standardize
    _tok = re.compile(r"[a-z']+")
    def english_only(s):
        return " ".join(w for w in _tok.findall(s.lower()) if w in english)
    def _cer(a, b):
        if not a.strip(): return 0.0
        if not b.strip(): return 1.0
        return float(jiwer.cer(a, b, reference_transform=cer_tx, hypothesis_transform=cer_tx))
    def score(ref, hyp):
        if not hyp.strip():
            return {"cer_raw":1.0,"cer_core":1.0,"wer":1.0,"sanskrit_load":0.0}
        cer_raw  = _cer(ref, hyp)
        cer_core = _cer(english_only(ref), english_only(hyp))
        try:
            wer = float(jiwer.wer(ref, hyp, reference_transform=wer_tx, hypothesis_transform=wer_tx))
        except ValueError:
            wer = 1.0
        return {"cer_raw":round(cer_raw,4),"cer_core":round(cer_core,4),
                "wer":round(wer,4),"sanskrit_load":round(cer_raw-cer_core,4)}
    return score


def run_qa(d: Path, asr, score, cfg):
    rows = [json.loads(l) for l in open(d / "train.jsonl", encoding="utf-8") if l.strip()]
    rep, passed = [], []
    print(f"  [qa] scoring {len(rows)} clips (align-confidence + silence{' + ASR-CER advisory' if asr else ''}) ...")
    for i, r in enumerate(rows, 1):
        wav = d / Path(r["audio"]).as_posix(); reasons = []
        amin = float(r.get("align_min", 0.0)); amean = float(r.get("align_mean", 0.0))
        m = {"cer_raw":-1.0,"cer_core":-1.0,"wer":-1.0,"sanskrit_load":-1.0}
        if not wav.exists():
            rep.append({"clip":i,"dur":r.get("duration"),"align_min":amin,"align_mean":amean,**m,
                        "lead":-1.0,"trail":-1.0,"status":"FAIL","reasons":"missing_wav"}); continue
        # PRIMARY gate: per-clip alignment confidence (immune to Whisper Sanskrit-misreads + hallucination)
        if amin < cfg["min_align"]: reasons.append(f"low_align({amin:.2f})")
        # silence edges
        y, sr = librosa.load(str(wav), sr=16000); se = silence_edges(y, sr)
        if se is None:
            reasons.append("all_silence"); lead = trail = -1.0
        else:
            lead, trail = round(se[0],2), round(se[1],2)
            if se[0] > cfg["max_lead_sil"]:  reasons.append(f"lead_sil({se[0]:.2f})")
            if se[1] > cfg["max_trail_sil"]: reasons.append(f"trail_sil({se[1]:.2f})")
        # ASR-CER (ADVISORY; only an EGREGIOUS cer_core flags a clip — Whisper is noisy on Sanskrit)
        if asr is not None:
            segs, _ = asr.transcribe(str(wav), language="en", beam_size=5, vad_filter=True,
                                     vad_parameters=dict(min_silence_duration_ms=500), condition_on_previous_text=False)
            m = score(r["text"], " ".join(s.text.strip() for s in segs).strip())
            if m["cer_core"] > cfg["max_cer_core"]: reasons.append(f"high_cer_core({m['cer_core']:.2f})")
        ok = not reasons
        rep.append({"clip":i,"dur":r.get("duration"),"align_min":round(amin,3),"align_mean":round(amean,3),
                    **m,"lead":lead,"trail":trail,"status":"PASS" if ok else "FAIL","reasons":";".join(reasons)})
        if ok: passed.append(r)
    (d / "train.clean.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in passed), encoding="utf-8")
    cols = ["clip","dur","align_min","align_mean","cer_raw","cer_core","wer","sanskrit_load","lead","trail","status","reasons"]
    with (d / "qa_asr_report.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for x in rep: f.write("\t".join(str(x[c]) for c in cols) + "\n")
    def dist(key, drop_neg):
        v = sorted(float(x[key]) for x in rep if not (drop_neg and float(x.get(key, -1)) < -0.5))
        if not v: return "n/a"
        q = lambda p: float(np.percentile(v, p))
        return f"min={v[0]:.3f}  p10={q(10):.3f}  p50={q(50):.3f}  p90={q(90):.3f}  max={v[-1]:.3f}"
    print("  [qa] distributions (calibrate --min-align / --max-cer-core from these):")
    for k in ("align_min","align_mean","cer_core","cer_raw","wer"):
        print(f"        {k:11s} {dist(k, drop_neg=not k.startswith('align'))}")
    return rep, passed


# ===================================================================================
# one lecture, end to end
# ===================================================================================
def process_lecture(slug, models, cfg):
    diar_pipe, embed, cosine, centroid, align, asr, score = models
    d = LECT / slug
    assert (d / "audio.mp3").exists() and (d / "transcript.txt").exists(), \
        f"missing audio.mp3 / transcript.txt in {d} (run scripts/fetch_lecture.py first)"
    print(f"\n========== {slug} ==========")
    wav16 = to_wav_16k(d / "audio.mp3", ROOT / "_work" / f"{slug}.wav")

    turns, secs = diarize(diar_pipe, str(wav16), cfg["min_speakers"], cfg["max_speakers"])
    print(f"  [diarize] {len(turns)} turns, {len(secs)} speakers")
    intervals, flags = verify_ggs(turns, secs, wav16, cfg["ref_clips"], embed, cosine, centroid, cfg)

    raw = (d / "transcript.txt").read_text(encoding="utf-8")
    txt = transliterate(ggs_only_text(raw, cfg["speaker_re"], cfg["ggs_re"]).replace("\n", " "))
    words = align(str(wav16), txt, language=cfg["language"], batch_size=8)
    print(f"  [align] {len(words)} words")

    for w in words:
        w["is_ggs"] = word_is_ggs(w, intervals, cfg["overlap_frac"])
    nkeep = sum(w["is_ggs"] for w in words); dfrac = 1 - nkeep / max(1, len(words))
    print(f"  [intersect] kept {nkeep}/{len(words)} GGS words (drop {100*dfrac:.1f}%)")
    if dfrac > cfg["max_word_drop"]:
        flags.append(f"high_word_drop({dfrac:.2f})")

    rows = emit_clips(words, intervals, wav16, d / "clips", slug, cfg["min_sec"], cfg["max_sec"])
    (d / "train.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"  [cut] {len(rows)} clips")

    rep, passed = run_qa(d, asr, score, cfg)
    print(f"  [qa] PASS {len(passed)} / FAIL {sum(x['status']=='FAIL' for x in rep)}")
    if flags:
        print(f"  !! FLAGGED for review: {flags}")
    return len(passed), flags


# ===================================================================================
# group split (whole-lecture holdout) + ref_audio injection
# ===================================================================================
def group_split(out_dir, n_val_holdout=1, ref_frac=0.40, seed=1337):
    import random
    rng = random.Random(seed)
    by_lec = {}
    for d in sorted(p for p in LECT.iterdir() if p.is_dir()):
        m = d / "train.clean.jsonl"
        if not m.exists(): continue
        rows = []
        for line in m.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            r["audio"] = f"{d.name}/clips/{Path(r['audio']).name}"
            r["lecture"] = d.name
            rows.append(r)
        if rows: by_lec[d.name] = rows
    lectures = sorted(by_lec)
    if len(lectures) < 2:
        print(f"!! only {len(lectures)} lecture(s) with clean clips — group val needs >=2; val may be empty.")
    k = min(n_val_holdout, max(0, len(lectures) - 1)) or (1 if lectures else 0)
    val_lecs = set(rng.sample(lectures, k)) if lectures else set()
    train = [r for l in lectures if l not in val_lecs for r in by_lec[l]]
    val   = [r for l in val_lecs for r in by_lec[l]]
    train_audio = [r["audio"] for r in train]
    for i in (rng.sample(range(len(train)), int(round(ref_frac*len(train)))) if train else []):
        cand = train[i]["audio"]; ref = cand
        while ref == cand and len(train_audio) > 1: ref = rng.choice(train_audio)
        train[i]["ref_audio"] = ref
    assert set(r["lecture"] for r in train).isdisjoint(set(r["lecture"] for r in val)), \
        "LEAKAGE: a lecture appears in both train and val — whole-lecture holdout violated"
    assert all(r.get("ref_audio") != r["audio"] for r in train), \
        "BUG: ref_audio points at its own clip"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    def _w(rs, p):
        with open(p, "w", encoding="utf-8") as f:
            for r in rs: f.write(json.dumps({k:v for k,v in r.items() if k!="lecture"}, ensure_ascii=False)+"\n")
    _w(train, out_dir/"train.jsonl"); _w(val, out_dir/"val.jsonl")
    print(f"VAL (whole-lecture holdout): {sorted(val_lecs)} | TRAIN lectures: {[l for l in lectures if l not in val_lecs]}")
    print(f"train.jsonl: {len(train)} clips ({sum('ref_audio' in r for r in train)} with ref_audio) | val.jsonl: {len(val)} clips")


# ===================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="i-and-mine-and-namabhasa-stage")
    ap.add_argument("--scale", action="store_true", help="process all lectures + group split")
    ap.add_argument("--ref", nargs="+", required=True, help="clean GGS-only reference clip(s) for ECAPA")
    ap.add_argument("--language", default="eng")
    ap.add_argument("--min-sec", type=float, default=9.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=4)
    ap.add_argument("--cosine-thr", type=float, default=0.40)
    ap.add_argument("--cosine-margin", type=float, default=0.10)
    ap.add_argument("--min-turn-sec", type=float, default=1.0)
    ap.add_argument("--overlap-frac", type=float, default=0.9)
    ap.add_argument("--max-word-drop", type=float, default=0.15)
    ap.add_argument("--max-cer-core", type=float, default=0.85, help="ADVISORY: only egregious English-backbone CER flags")
    ap.add_argument("--max-wer", type=float, default=0.60)
    ap.add_argument("--min-align", type=float, default=-1e9, help="min per-clip alignment confidence gate (calibrate from dist)")
    ap.add_argument("--no-cer", action="store_true", help="skip Whisper ASR-CER (faster; align-confidence is the real gate)")
    ap.add_argument("--max-lead-sil", type=float, default=0.6)
    ap.add_argument("--max-trail-sil", type=float, default=1.0)
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--out", default=str(ROOT / "data" / "out"))
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:                                    # fallback: a token FILE (keeps the secret off the CLI/chat)
        for p in ("/workspace/.hf_token", str(Path.home() / ".hf_token")):
            if Path(p).exists():
                hf_token = Path(p).read_text(encoding="utf-8").strip()
                break
    if not hf_token:
        sys.exit("No HF token. Either `export HF_TOKEN=...` or write it to /workspace/.hf_token "
                 "(and accept the pyannote community-1 gated repos first).")
    for p in args.ref:
        if not Path(p).exists():
            sys.exit(f"reference clip not found: {p}")

    import torch, glob
    # Expose torch's bundled cuDNN-9 libs on LD_LIBRARY_PATH BEFORE faster-whisper/ctranslate2 load,
    # else Stage F crashes with "Unable to load libcudnn_ops.so.9" ~30 min into the run. The dynamic
    # loader re-reads LD_LIBRARY_PATH at dlopen time, so setting it here (pre-import) takes effect.
    _cudnn = sorted({os.path.dirname(p) for p in glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "cudnn", "lib", "*.so*"))})
    if _cudnn:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(_cudnn + [os.environ.get("LD_LIBRARY_PATH", "")])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    cfg = dict(ref_clips=[str(p) for p in args.ref], language=args.language,
               min_sec=args.min_sec, max_sec=args.max_sec,
               min_speakers=args.min_speakers, max_speakers=args.max_speakers,
               cosine_thr=args.cosine_thr, cosine_margin=args.cosine_margin, min_turn_sec=args.min_turn_sec,
               overlap_frac=args.overlap_frac, max_word_drop=args.max_word_drop,
               max_cer_core=args.max_cer_core, max_wer=args.max_wer, min_align=args.min_align,
               max_lead_sil=args.max_lead_sil, max_trail_sil=args.max_trail_sil,
               speaker_re=None, ggs_re=None)

    print("loading models (diarizer, ECAPA, aligner%s) ..." % ("" if args.no_cer else ", ASR"))
    diar_pipe = load_diar_pipeline(hf_token, device)
    embed, cosine, centroid = make_embedder(device)
    align = make_aligner(device)
    asr = None
    if not args.no_cer:
        from faster_whisper import WhisperModel
        asr = WhisperModel(args.asr_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    score = make_scorer(load_english())
    models = (diar_pipe, embed, cosine, centroid, align, asr, score)

    slugs = [p.name for p in sorted(LECT.iterdir()) if p.is_dir()] if args.scale else [args.slug]
    flagged = {}
    for s in slugs:
        try:
            _, flags = process_lecture(s, models, cfg)
            if flags: flagged[s] = flags
        except Exception as e:
            print(f"  !! {s} FAILED: {e}")
            flagged[s] = [f"error:{e}"]
    if args.scale:
        group_split(args.out)
    if flagged:
        print("\n!! REVIEW FLAGGED:")
        for s, f in flagged.items(): print(f"   {s}: {f}")


if __name__ == "__main__":
    main()
