#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whisper-small x CDSD, three seeds: micro + macro bootstrap and exact sign-flip.

Completes the seed-balanced 2x2 behind the CDSD cross-model reversal. Reports all three estimators for
every contrast, because on the Qwen half of this comparison the micro and macro answers differ for
`upper` - `all` and that difference is itself a reported finding.
"""
import itertools, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

PRED = Path.home() / "dysarthria" / "results" / "probe" / "whisper_cdsd_multiseed_preds"
S = [42, 43, 44]
BANDS = ["lower", "middle", "upper", "all"]
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


def tally(path):
    acc = defaultdict(lambda: [0, 0])
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        acc[r["speaker_id"]][0] += _edit(list(_P.sub("", r["ref"] or "")),
                                         list(_P.sub("", r["hyp"] or "")))
        acc[r["speaker_id"]][1] += len(list(_P.sub("", r["ref"] or "")))
    return acc


def micro(sel, d):
    e = sum(d[s][0] for s in sel if s in d)
    n = sum(d[s][1] for s in sel if s in d)
    return 100.0 * e / max(n, 1)


def macro(sel, d):
    return float(np.mean([100.0 * d[s][0] / max(d[s][1], 1) for s in sel if s in d]))


base = tally(PRED / "s42" / "base.test.preds.jsonl")
arms = {(b, s): tally(PRED / f"s{s}" / f"{b}_s{s}.test.preds.jsonl") for s in S for b in BANDS}
spk = sorted(base)
print(f"speakers={spk}  seeds={S}\n")

print("per-variant CER (micro, mean +/- SD over seeds)")
print(f"  {'base':7s} {micro(spk, base):6.3f}  (single run)")
for b in ["all", "middle", "upper", "lower"]:
    v = [micro(spk, arms[(b, s)]) for s in S]
    print(f"  {b:7s} {np.mean(v):6.3f} +/- {np.std(v, ddof=1):.3f}   per-seed {[round(x,3) for x in v]}")

rng = np.random.default_rng(42)
idx = rng.integers(0, len(spk), size=(B, len(spk)))
pick = rng.integers(0, len(S), size=B)
PAIRS = [("all", "upper"), ("all", "middle"), ("all", "lower"), ("middle", "upper")]


def boot(a, b, fn):
    per_seed = [fn(spk, arms[(a, s)]) - fn(spk, arms[(b, s)]) for s in S]
    m = float(np.mean(per_seed))
    draws = np.empty(B)
    for i in range(B):
        sel = [spk[j] for j in idx[i]]
        s = S[pick[i]]
        draws[i] = fn(sel, arms[(a, s)]) - fn(sel, arms[(b, s)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return m, float(np.std(per_seed, ddof=1)), lo, hi, (lo > 0) == (hi > 0)


def signflip(a, b):
    d = np.array([np.mean([100.0 * arms[(a, s)][k][0] / max(arms[(a, s)][k][1], 1)
                           - 100.0 * arms[(b, s)][k][0] / max(arms[(b, s)][k][1], 1)
                           for s in S]) for k in spk])
    obs = abs(d.mean())
    cnt = sum(1 for sg in itertools.product([1, -1], repeat=len(d))
              if abs((d * np.array(sg)).mean()) >= obs - 1e-12)
    return cnt / 2 ** len(d), d


print(f"\n{'contrast':>18} {'MICRO mean':>11} {'95% CI':>18} {'':>10}  {'MACRO mean':>11} {'95% CI':>18} {'':>10}  exact p")
for a, b in PAIRS:
    m1, sd1, lo1, hi1, r1 = boot(a, b, micro)
    m2, sd2, lo2, hi2, r2 = boot(a, b, macro)
    p, d = signflip(a, b)
    print(f"{a+' - '+b:>18} {m1:+11.3f} [{lo1:+7.2f},{hi1:+7.2f}] {'RESOLVED' if r1 else 'spans 0':>10}  "
          f"{m2:+11.3f} [{lo2:+7.2f},{hi2:+7.2f}] {'RESOLVED' if r2 else 'spans 0':>10}  {p:.4f}")
    print(f"{'':18} seedSD micro {sd1:.3f}   per-speaker macro diffs {[round(x,2) for x in d]}")
