#!/usr/bin/env python3
"""Decode the probe set to produce Probe C's recognition-difficulty labels.

Muthu et al. define difficulty by decoding each utterance with a STRONGER external model than the one
being probed (they probe Whisper-small and label with Whisper-large), computing per-utterance CER, and
binning into low/medium/high by the 33.33rd and 66.67th percentiles.

CANDOR labels with **base `Qwen/Qwen3-ASR-1.7B`**, NOT the stronger `qwen3-asr-1.7b-dys-zh-aug`.
zh-aug was fine-tuned on CDSD and has already seen **81.9% of the probe utterances and 37 of 44
speakers, including all 6 held-out test speakers** — so its per-utterance CER measures memorisation, not
difficulty. Base has never seen CDSD, so its CER is a genuine difficulty signal.

This deviates from Muthu, who label with a model STRONGER than the probed one (Whisper-large labelling
Whisper-small). No uncontaminated stronger Mandarin model is available to us, so we trade strength for
cleanliness and say so. Difficulty here means "hard for a general-purpose ASR model", which is arguably
the better construct anyway.

Writes results/probe/cdsd_difficulty_labels.jsonl with per-utterance CER and tertile.

**Runs in the MAIN venv, not venv_qwen3asr.** Decoding goes through Qwen's own `qwen_asr` library
(`Qwen3ASRModel.transcribe`), which is the path every published Track 1 number uses, and which is
installed only in the main venv (transformers 4.57.6 — `qwen_asr` bundles its own modelling code, so it
does not need transformers' native qwen3_asr support). A generic
`processor.apply_chat_template` + `model.generate` path was tried first and is badly degraded: on 369
utterances that overlap the recorded fair-eval predictions it scored **124.57% CER against the recorded
41.87%** for the same model and utterances. Do not reintroduce it.

Usage (DGX, MAIN venv):
    source venv/bin/activate
    python scripts/layerprobe/decode_for_difficulty.py \
        --model "$DYS_ROOT"/models/qwen3-asr-zh-aug/ep2/final
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
SR = 16000
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")


def norm(text: str) -> str:
    return _PUNCT.sub("", text or "")


def cer(ref: str, hyp: str) -> float:
    r, h = list(norm(ref)), list(norm(hyp))
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, cr in enumerate(r, 1):
        cur = [i]
        for j, ch in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cr != ch)))
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_probe_manifest.jsonl")
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "cdsd_difficulty_labels.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--language", default="Chinese")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    done: dict[str, dict] = {}
    if args.out.exists():
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["uid"]] = rec
        print(f"resuming: {len(done)} already decoded")

    from qwen_asr import Qwen3ASRModel

    print(f"loading {args.model} via qwen_asr ...")
    asr = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )

    todo = [r for r in rows if r["uid"] not in done]
    print(f"decoding {len(todo)} utterances (batch {args.batch_size}) ...")
    t0 = time.time()
    out_fh = open(args.out, "a", encoding="utf-8")
    n = 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        paths = [r["audio_filepath"] for r in chunk]
        try:
            outs = asr.transcribe(audio=paths, language=args.language)
            texts = [(o.text if o is not None else "") for o in outs]
        except Exception as exc:  # noqa: BLE001
            print(f"  !! batch at {i}: {type(exc).__name__}: {exc} — retrying singly")
            texts = []
            for r in chunk:
                try:
                    o = asr.transcribe(audio=r["audio_filepath"], language=args.language)
                    texts.append(o[0].text if o else "")
                except Exception as exc2:  # noqa: BLE001
                    print(f"     !! {r['uid']}: {type(exc2).__name__}: {exc2}")
                    texts.append("")
        for rec, text in zip(chunk, texts):
            if "<asr_text>" in text:
                text = text.split("<asr_text>", 1)[1]
            row = {"uid": rec["uid"], "speaker_id": rec["speaker_id"], "split": rec["split"],
                   "ref": rec["text"], "hyp": text, "cer": round(cer(rec["text"], text), 4)}
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[rec["uid"]] = row
            n += 1
        out_fh.flush()
        if i % (args.batch_size * 20) == 0:
            print(f"  {n}/{len(todo)}  {n/max(time.time()-t0,1e-6):.1f} utt/s  "
                  f"running CER={100*np.mean([d['cer'] for d in done.values()]):.2f}%")
    out_fh.close()

    # Difficulty tiers. Muthu use plain 33.33 / 66.67 percentile thresholds, which degenerate here:
    # the base model's CER distribution has two large mass points — 2,267 utterances at exactly 0.0
    # (perfect) and 2,853 at exactly 1.0 (nothing recovered) — so the 66.67th percentile lands ON 1.0
    # and everything equal to it collapses into one tier (observed split 2943 / 5503 / 354).
    # Binning on the natural breakpoints instead keeps the tiers balanced AND never splits a tie mass
    # across two labels.
    vals = np.array([done[r["uid"]]["cer"] for r in rows if r["uid"] in done])
    lo = float(np.percentile(vals, 100 / 3))
    labelled = []
    for r in rows:
        d = done.get(r["uid"])
        if d is None:
            continue
        if d["cer"] >= 1.0:
            tier = "high"          # the decoder recovered nothing usable
        elif d["cer"] < lo:
            tier = "low"
        else:
            tier = "medium"
        labelled.append({**d, "difficulty": tier})
    with open(args.out, "w", encoding="utf-8") as fh:
        for d in labelled:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    from collections import Counter

    print(f"\ndecoded {len(labelled)} utterances, mean CER {100*vals.mean():.2f}%")
    print(f"tertile thresholds: low <= {lo:.4f} < medium <= {hi:.4f} < high")
    print(f"difficulty distribution: {dict(Counter(d['difficulty'] for d in labelled))}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
