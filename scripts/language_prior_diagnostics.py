#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify the shallow-adaptation "language-prior takeover" (Paper 10a §7.2).

The claim is qualitative in the current draft: shallow LoRA makes the model emit fluent, corpus-typical
text unrelated to the audio. Two cheap diagnostics make it measurable from predictions we already have:

  length ratio  = |hyp| / |ref|          (>1 means the model emits more than was said)
  insertion rate = max(0, |hyp|-|ref|) / |ref|

Both averaged per speaker over three seeds. The decoder is frozen throughout, so any change is caused by
altering the encoder representations feeding it, not by changed language-model weights.
"""
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

PRED = Path.home() / "dysarthria" / "results" / "probe" / "cdsd_multiseed_preds"
S = [42, 43, 44]
_P = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")


def stats(path):
    d = defaultdict(lambda: [0, 0])
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ref = len(_P.sub("", r["ref"] or ""))
        hyp = len(_P.sub("", r["hyp"] or ""))
        d[r["speaker_id"]][0] += hyp
        d[r["speaker_id"]][1] += ref
    return {s: (h, n) for s, (h, n) in d.items()}


base = stats(PRED / "s42" / "base.test.preds.jsonl")
arms = {b: [stats(PRED / f"s{s}" / f"{b}_s{s}.test.preds.jsonl") for s in S]
        for b in ("lower", "middle", "upper", "all")}
spk = sorted(base)

print("length ratio |hyp|/|ref|, per speaker (base is a single deterministic run)")
print(f"{'speaker':10s} {'base':>7s} {'lower':>7s} {'middle':>7s} {'upper':>7s} {'all':>7s}")
for s in spk:
    row = f"{s:10s} {base[s][0]/max(base[s][1],1):7.3f}"
    for b in ("lower", "middle", "upper", "all"):
        v = np.mean([a[s][0] / max(a[s][1], 1) for a in arms[b]])
        row += f" {v:7.3f}"
    print(row)

print("\ncorpus-level insertion rate  max(0,|hyp|-|ref|)/|ref|")
tot_ref = sum(base[s][1] for s in spk)
print(f"  {'base':7s} {max(0, sum(base[s][0] for s in spk) - tot_ref)/tot_ref:7.4f}")
for b in ("lower", "middle", "upper", "all"):
    v = np.mean([max(0, sum(a[s][0] for s in spk) - tot_ref) / tot_ref for a in arms[b]])
    print(f"  {b:7s} {v:7.4f}")
