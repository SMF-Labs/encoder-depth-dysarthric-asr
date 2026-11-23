#!/usr/bin/env python3
"""Is Whisper-small's 103.57% CDSD CER a model failure or a decode-configuration artefact?

WHY THIS MATTERS. The paper's cross-model contrast compares a working model (Qwen3-ASR, 32.06% CER
unadapted) against one scoring above 100% — where insertions exceed reference length, the signature of
hallucination rather than degraded transcription. A reviewer flagged that this makes the comparison
mechanistically ambiguous: all-block LoRA may win for Whisper because a broken model needs rescuing
everywhere, not because anything architectural favours distributed adaptation.

Before accepting that confound, it is worth asking whether OUR decode settings caused part of it. The
evaluator generates with max_new_tokens=200 and greedy search, with no repetition control. CDSD
utterances average ~2.4 s, which is perhaps 10-20 Mandarin characters. A 200-token budget leaves ample
room for a repetition loop, and every looped character is an insertion.

This scores the unadapted Whisper-small on a fixed subset under several decode configurations. If a
better-configured decode moves the baseline substantially, part of the "degenerate baseline" is ours and
the paper must say so. If it does not, the confound is real and stays in the limitations as such.

Runs on the base model only - no adapters, nothing to merge.

Usage:
    python scripts/layerprobe/whisper_decode_check.py --limit 400
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE = Path(__file__).resolve().parents[2]
PROBE = BASE / "results" / "probe"
SR = 16000
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")


def norm(t: str) -> str:
    return _PUNCT.sub("", t or "")


def edit(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_audio(path: str) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        import torchaudio
        a = (torchaudio.transforms.Resample(sr, SR)(torch.from_numpy(a).unsqueeze(0))
             .squeeze(0).numpy())
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROBE / "cdsd_lora_manifest.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--out", type=Path, default=PROBE / "whisper_decode_check.csv")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == args.split]
    rows.sort(key=lambda r: r["uid"])          # deterministic subset, not a random one
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} utts from {args.split}; mean duration "
          f"{np.mean([r.get('duration', 0) for r in rows]):.2f}s")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to("cuda").eval()
    forced = proc.get_decoder_prompt_ids(language="chinese", task="transcribe")

    # The published setting first, then one change at a time so any improvement is attributable.
    configs = [
        ("published (max200, greedy)", dict(max_new_tokens=200)),
        ("cap 64 tokens", dict(max_new_tokens=64)),
        ("cap 48 + no_repeat_3", dict(max_new_tokens=48, no_repeat_ngram_size=3)),
        ("cap 48 + rep_penalty 1.2", dict(max_new_tokens=48, repetition_penalty=1.2)),
        ("cap 48 + no_repeat_3 + beam 5", dict(max_new_tokens=48, no_repeat_ngram_size=3, num_beams=5)),
    ]

    results = []
    for name, kw in configs:
        ce = ct = ins = 0
        t0 = time.time()
        for r in rows:
            try:
                audio = read_audio(r["audio_filepath"])
            except Exception:
                continue
            feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features.to("cuda")
            with torch.no_grad():
                ids = model.generate(feats, forced_decoder_ids=forced, **kw)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
            ref_c, hyp_c = list(norm(r["text"])), list(norm(hyp))
            ce += edit(ref_c, hyp_c)
            ct += len(ref_c)
            ins += max(0, len(hyp_c) - len(ref_c))     # crude over-generation measure
        cer = 100.0 * ce / max(ct, 1)
        over = 100.0 * ins / max(ct, 1)
        print(f"  {name:34s} CER {cer:7.2f}   over-generation {over:6.2f}%   [{time.time()-t0:.0f}s]")
        results.append({"config": name, "cer": round(cer, 3),
                        "over_generation_pct": round(over, 3), "n": len(rows)})

    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    best = min(results, key=lambda r: r["cer"])
    print(f"\npublished config: {results[0]['cer']:.2f} CER")
    print(f"best config     : {best['cer']:.2f} CER ({best['config']})")
    print(f"difference      : {results[0]['cer'] - best['cer']:+.2f} pp")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
