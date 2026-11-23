#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-speaker-out micro CER over the three-seed CDSD sweep (Paper 10a Table 13)."""
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np

PRED = Path.home() / "dysarthria" / "results" / "probe" / "cdsd_multiseed_preds"
SEEDS = [42, 43, 44]
_P = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")


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
        ref = list(_P.sub("", r["ref"] or ""))
        hyp = list(_P.sub("", r["hyp"] or ""))
        acc[r["speaker_id"]][0] += _edit(ref, hyp)
        acc[r["speaker_id"]][1] += len(ref)
    return acc


base = tally(PRED / "s42" / "base.test.preds.jsonl")
arms = {(b, s): tally(PRED / f"s{s}" / f"{b}_s{s}.test.preds.jsonl")
        for s in SEEDS for b in ("upper", "middle", "all")}
spk = sorted(base)


def micro(d, excl=None):
    e = sum(v[0] for k, v in d.items() if k != excl)
    n = sum(v[1] for k, v in d.items() if k != excl)
    return 100.0 * e / max(n, 1)


def f(v):
    return "%.2f ± %.2f" % (st.mean(v), st.stdev(v)) if len(v) > 1 else "%.2f" % v[0]


print("| Excluded | base | `upper` | `middle` | `all` | **`upper`−base** | **`upper`−`middle`** |")
print("|---|---|---|---|---|---|---|")
for ex in [None] + spk:
    b = micro(base, ex)
    row = {}
    for band in ("upper", "middle", "all"):
        row[band] = [micro(arms[(band, s)], ex) for s in SEEDS]
    ub = [row["upper"][i] - b for i in range(3)]
    um = [row["upper"][i] - row["middle"][i] for i in range(3)]
    label = "none" if ex is None else ex
    print("| %s | %.2f | %s | %s | %s | %s | %s |" % (
        label, b, f(row["upper"]), f(row["middle"]), f(row["all"]), f(ub), f(um)))
