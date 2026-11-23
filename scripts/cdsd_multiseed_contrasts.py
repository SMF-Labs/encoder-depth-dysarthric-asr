#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speaker x seed bootstrap for the CDSD x Qwen multi-seed band sweep (Paper 10a, review pass 5).

The published CDSD intervals resampled the six test SPEAKERS from a SINGLE training run, so they
quantified test-speaker sampling and nothing about optimisation stochasticity. This uses the same
estimator as the UASpeech tables: resample speakers with replacement AND draw a seed, forming each
contrast WITHIN a seed so that a seed's overall difficulty cancels.
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

PROBE = Path.home() / "dysarthria" / "results" / "probe"
PRED = PROBE / "cdsd_multiseed_preds"
SEEDS = [42, 43, 44]
BANDS = ["lower", "middle", "upper", "all"]
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")
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


def tally(path):
    acc = defaultdict(lambda: ([], []))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ref = list(_PUNCT.sub("", r["ref"] or ""))
            hyp = list(_PUNCT.sub("", r["hyp"] or ""))
            acc[r["speaker_id"]][0].append(_edit(ref, hyp))
            acc[r["speaker_id"]][1].append(len(ref))
    return {s: (np.array(e), np.array(n)) for s, (e, n) in acc.items()}


def cer(sel, d):
    e = sum(d[s][0].sum() for s in sel if s in d)
    n = sum(d[s][1].sum() for s in sel if s in d)
    return 100.0 * e / max(n, 1)


base = None
arms = {}
for sd in SEEDS:
    p = PRED / f"s{sd}" / "base.test.preds.jsonl"
    if base is None and p.exists():
        base = tally(p)
    for b in BANDS:
        f = PRED / f"s{sd}" / f"{b}_s{sd}.test.preds.jsonl"
        if not f.exists():
            print(f"MISSING {f}"); sys.exit(1)
        arms[(b, sd)] = tally(f)
spk = sorted(base)
print(f"speakers={spk}  seeds={SEEDS}\n")


def arm(b, sd):
    return base if b == "base" else arms[(b, sd)]


print("per-variant micro CER")
print(f"  {'base':7s} {cer(spk, base):6.3f}   (single run, no training)")
for b in ["upper", "middle", "all", "lower"]:
    v = [cer(spk, arms[(b, s)]) for s in SEEDS]
    print(f"  {b:7s} {np.mean(v):6.3f} +/- {np.std(v, ddof=1):.3f}   per-seed {[round(x,3) for x in v]}")

rng = np.random.default_rng(42)
idx = rng.integers(0, len(spk), size=(B, len(spk)))
pick = rng.integers(0, len(SEEDS), size=B)

PAIRS = [("lower", "middle"), ("upper", "middle"), ("upper", "all"), ("middle", "all"), ("upper", "lower"),
         ("upper", "base"), ("all", "base"), ("middle", "base"), ("lower", "base")]
print(f"\nspeaker x seed bootstrap, B={B:,} (contrasts formed within a seed)")
print(f"{'contrast':>18} {'mean':>8} {'seedSD':>7}   {'95% CI':>18}  verdict")
for a, b in PAIRS:
    per_seed = [cer(spk, arm(a, s)) - cer(spk, arm(b, s)) for s in SEEDS]
    m = float(np.mean(per_seed))
    sd_ = float(np.std(per_seed, ddof=1))
    draws = np.empty(B)
    for i in range(B):
        sel = [spk[j] for j in idx[i]]
        s = SEEDS[pick[i]]
        draws[i] = cer(sel, arm(a, s)) - cer(sel, arm(b, s))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    excl = (lo > 0) == (hi > 0)
    print(f"{a+' - '+b:>18} {m:+8.3f} {sd_:7.3f}   [{lo:+7.2f}, {hi:+7.2f}]  "
          f"{'RESOLVED' if excl else 'spans zero'}")
