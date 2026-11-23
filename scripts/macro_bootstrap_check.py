#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the bootstrap/sign-flip disagreement about the ESTIMATOR or about the ESTIMAND?

The paper's bootstrap pools edit counts over resampled speakers (MICRO CER), which weights a speaker by
their reference length. The exact sign-flip test operates on per-speaker CER (MACRO), weighting speakers
equally. Those are different quantities, so a disagreement between them is not automatically evidence
that the bootstrap is unreliable. This runs a MACRO speaker x seed bootstrap so the comparison against
the exact test is like-for-like.
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

PROBE = Path.home() / "dysarthria" / "results" / "probe"
S = [42, 43, 44]
_P = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")
B = 10000


def _edit(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def per_speaker(path):
    acc = defaultdict(lambda: [0, 0])
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        acc[r["speaker_id"]][0] += _edit(list(_P.sub("", r["ref"] or "")),
                                         list(_P.sub("", r["hyp"] or "")))
        acc[r["speaker_id"]][1] += len(list(_P.sub("", r["ref"] or "")))
    return {s: 100.0 * e / max(n, 1) for s, (e, n) in acc.items()}


ms = PROBE / "cdsd_multiseed_preds"
arms = {(b, s): per_speaker(ms / f"s{s}" / f"{b}_s{s}.test.preds.jsonl")
        for s in S for b in ("lower", "middle", "upper", "all")}
spk = sorted(arms[("upper", 42)])
rng = np.random.default_rng(42)
idx = rng.integers(0, len(spk), size=(B, len(spk)))
pick = rng.integers(0, len(S), size=B)

print("MACRO (speaker-equal-weight) speaker x seed bootstrap, B=10,000")
print(f"{'contrast':>18} {'mean':>8}   {'95% CI':>18}  verdict")
for a, b in (("upper", "all"), ("upper", "middle"), ("upper", "lower"), ("middle", "all")):
    per_seed = [np.mean([arms[(a, s)][k] - arms[(b, s)][k] for k in spk]) for s in S]
    m = float(np.mean(per_seed))
    draws = np.empty(B)
    for i in range(B):
        sel = [spk[j] for j in idx[i]]
        s = S[pick[i]]
        draws[i] = np.mean([arms[(a, s)][k] - arms[(b, s)][k] for k in sel])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    excl = (lo > 0) == (hi > 0)
    print(f"{a+' - '+b:>18} {m:+8.3f}   [{lo:+7.2f}, {hi:+7.2f}]  "
          f"{'RESOLVED' if excl else 'spans zero'}")
