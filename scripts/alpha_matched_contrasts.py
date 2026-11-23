#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speaker x seed bootstrap for the ALPHA-MATCHED placement x capacity 2x2 (Paper 10a §6.7).

Every arm here has alpha = 2r, so effective LoRA scaling is constant across the design and the two
remaining differences between a matched pair are placement and per-block rank concentration. The
earlier 2x2 left alpha at 128 for every rank, which made alpha/r differ by 3x within the matched pair.
"""
import json, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

PRED = Path.home() / "dysarthria" / "results" / "probe" / "alpha_matched_preds"
SEEDS = [42, 43, 44]
CELLS = ["upper_r64", "all_r21", "upper_r192", "all_r64"]
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
    acc = defaultdict(lambda: ([], []))
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        acc[r["speaker_id"]][0].append(_edit(list(_P.sub("", r["ref"] or "")),
                                             list(_P.sub("", r["hyp"] or ""))))
        acc[r["speaker_id"]][1].append(len(list(_P.sub("", r["ref"] or ""))))
    return {s: (np.array(e), np.array(n)) for s, (e, n) in acc.items()}


def cer(sel, d):
    e = sum(d[s][0].sum() for s in sel if s in d)
    n = sum(d[s][1].sum() for s in sel if s in d)
    return 100.0 * e / max(n, 1)


base = tally(PRED / "s42" / "base.test.preds.jsonl")
arms = {}
for sd in SEEDS:
    for c in CELLS:
        f = PRED / f"s{sd}" / f"{c}_s{sd}.test.preds.jsonl"
        if not f.exists():
            print("MISSING", f); sys.exit(1)
        arms[(c, sd)] = tally(f)
spk = sorted(base)
print(f"speakers={spk}  seeds={SEEDS}\n")

print("alpha-matched 2x2 (alpha = 2r everywhere), micro CER")
print(r"  placement \ budget      ~4.2 M              12.58 M")
for pl, lo_c, hi_c in (("upper (8 blocks) ", "upper_r64", "upper_r192"),
                       ("all   (24 blocks)", "all_r21", "all_r64")):
    lo = [cer(spk, arms[(lo_c, s)]) for s in SEEDS]
    hi = [cer(spk, arms[(hi_c, s)]) for s in SEEDS]
    print(f"  {pl}   {np.mean(lo):6.3f} +/- {np.std(lo, ddof=1):.3f}   {np.mean(hi):6.3f} +/- {np.std(hi, ddof=1):.3f}")

rng = np.random.default_rng(42)
idx = rng.integers(0, len(spk), size=(B, len(spk)))
pick = rng.integers(0, len(SEEDS), size=B)

PAIRS = [("upper_r192", "all_r64"), ("upper_r64", "all_r21"),
         ("upper_r192", "upper_r64"), ("all_r64", "all_r21")]
print(f"\nspeaker x seed bootstrap, B={B:,} (contrasts formed within a seed)")
print(f"{'contrast':>26} {'mean':>8} {'seedSD':>7}   {'95% CI':>18}  verdict")
for a, b in PAIRS:
    per_seed = [cer(spk, arms[(a, s)]) - cer(spk, arms[(b, s)]) for s in SEEDS]
    m = float(np.mean(per_seed))
    sd_ = float(np.std(per_seed, ddof=1))
    draws = np.empty(B)
    for i in range(B):
        sel = [spk[j] for j in idx[i]]
        s = SEEDS[pick[i]]
        draws[i] = cer(sel, arms[(a, s)]) - cer(sel, arms[(b, s)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    excl = (lo > 0) == (hi > 0)
    print(f"{a+' - '+b:>26} {m:+8.3f} {sd_:7.3f}   [{lo:+7.2f}, {hi:+7.2f}]  "
          f"{'RESOLVED' if excl else 'spans zero'}")
