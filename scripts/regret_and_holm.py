#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selection regret for the three probe heuristics, plus a Holm-adjusted primary hypothesis family.

WHY REGRET. The paper currently scores a heuristic as failing if it does not name the empirical argmin.
That treats "missed the best band by 0.2 pp in a cell where the top two are statistically
indistinguishable" the same as "chose the lower band when it is 5 pp worse". Regret

    R(h) = CER[h's choice] - min_b CER[b]

says how much the choice actually costs, and a "compatible-with-best" set says whether the miss is even
resolvable. Both are more informative than a win/lose label.

WHY HOLM. Many pairwise contrasts are reported. A small pre-nominated family of primary hypotheses is
adjusted here; everything else in the paper stays explicitly exploratory.

Whisper x CDSD is marked PROVISIONAL: it is single-seed until the three-seed run completes.
"""
import numpy as np

# Band CER per cell. Qwen x CDSD and both UASpeech cells are three-seed means.
CELLS = {
    "Qwen3-ASR x CDSD":      {"lower": 35.52, "middle": 30.71, "upper": 29.88, "all": 31.19},
    "Qwen3-ASR x UASpeech":  {"lower": 52.02, "middle": 37.61, "upper": 34.19, "all": 32.66},
    "Whisper-small x CDSD":  {"lower": 56.21, "middle": 55.26, "upper": 57.03, "all": 53.54},
    "Whisper-small x UASpeech": {"lower": 51.34, "middle": 45.95, "upper": 44.42, "all": 40.41},
}
PROVISIONAL = {"Whisper-small x CDSD"}

# What each heuristic selects, from the measured probe curves (all four cells agree on these).
HEURISTIC = {
    "steepest PER improvement": "lower",
    "lowest band-mean PER":     "upper",
    "knee location":            "upper",
}

print("=" * 96)
print("SELECTION REGRET  R(h) = CER[choice] - CER[best]   (pp; 0.00 = picked the empirical best)")
print("=" * 96)
print(f"{'cell':28s} {'best':>7s} {'CER':>7s} | " + " | ".join(f"{h[:22]:>22s}" for h in HEURISTIC))
for cell, cer in CELLS.items():
    best = min(cer, key=cer.get)
    tag = " *" if cell in PROVISIONAL else ""
    row = f"{cell+tag:28s} {best:>7s} {cer[best]:7.2f} | "
    row += " | ".join(f"{HEURISTIC[h]:>6s} {cer[HEURISTIC[h]] - cer[best]:+6.2f} pp" for h in HEURISTIC)
    print(row)
print("\n* Whisper x CDSD is single-seed and PROVISIONAL pending the three-seed run.")

print("\nregret summary across the four cells")
for h, choice in HEURISTIC.items():
    r = [CELLS[c][choice] - min(CELLS[c].values()) for c in CELLS]
    hits = sum(1 for x in r if x == 0)
    print(f"  {h:26s} picks the best in {hits}/4 cells; regret "
          f"mean {np.mean(r):.2f} pp, worst {max(r):.2f} pp")

print("\n" + "=" * 96)
print("HOLM-ADJUSTED PRIMARY FAMILY (exact paired sign-flip p-values, speaker-level)")
print("=" * 96)
# Pre-nominated primary hypotheses: the probe-derived target vs the empirical alternative, the headline
# placement contrast, and the harmful-adaptation claim. Everything else stays exploratory.
FAMILY = [
    ("Qwen x CDSD:  upper - all      (headline placement)", 0.1562),
    ("Qwen x CDSD:  upper - lower    (probe target vs worst)", 0.0312),
    ("Whisper x UASpeech: all - upper (cross-model reversal)", 0.1250),
]
FAMILY.sort(key=lambda t: t[1])
m = len(FAMILY)
prev = 0.0
print(f"{'hypothesis':56s} {'raw p':>8s} {'Holm p':>8s}  verdict")
for i, (name, p) in enumerate(FAMILY):
    adj = max(prev, min(1.0, (m - i) * p))
    prev = adj
    print(f"{name:56s} {p:8.4f} {adj:8.4f}  {'SIG' if adj < 0.05 else 'n.s.'}")
print("\nNote the attainable floor: with 6 speakers the smallest two-sided exact p is 0.031, with 5 it is")
print("0.062. A unanimous 5-speaker contrast therefore CANNOT reach 0.05 by this test, so an n.s. result")
print("here is a statement about sample size as much as about effect.")
