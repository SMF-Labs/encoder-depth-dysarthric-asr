#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interaction test: do the two models actually DIFFER in preferred placement on CDSD?

WHY. Reporting that `upper` - `all` resolves negative on Qwen3-ASR and positive on Whisper-small does
not by itself establish an interaction: two separately-resolved intervals of opposite sign are not a
test of their difference. Because both models were evaluated on the SAME six held-out speakers, the
difference of differences

    DoD = (upper - all)_Qwen  -  (upper - all)_Whisper

can be formed inside one bootstrap replicate, with speakers resampled jointly and a seed drawn per model,
so speaker difficulty cancels on both sides.

Reported under micro CER (the paper's primary metric) and macro CER (speakers weighted equally), plus an
exact paired sign-flip over all 2^6 speaker assignments.

DATA. By default this reads results/cdsd_interaction_speaker_tallies.csv: for each
(model, band, seed, speaker), the total character edit distance and total reference length, both
measured after the punctuation stripping below. The test never uses an individual utterance -- micro and
macro CER are exact functions of those two integers -- so the released path reproduces the original
numbers exactly rather than approximately. This matters because Section 10 does not release transcripts
or per-utterance predictions, and without the tallies the paper's primary claim would be the one result
a reader could not check. Pass --from-preds to recompute from the prediction JSONLs instead, which
requires the working tree and the corpus.
"""
import argparse, csv, itertools, json, re
from collections import defaultdict
import os
from pathlib import Path
import numpy as np

# Results directory: this repository's own results/ by default, so the script runs from a clean
# clone. Set P10A_RESULTS to point at a working tree instead.
P = Path(os.environ.get("P10A_RESULTS",
                        Path(__file__).resolve().parent.parent / "results"))
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


def from_tallies(path):
    """Load the released per-speaker tallies into the same shape tally() returns."""
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["model"], r["band"], int(r["seed"]))][r["speaker_id"]] = \
                [int(r["edits"]), int(r["ref_chars"])]
    return out


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--from-preds", action="store_true",
                help="recompute from per-utterance prediction JSONLs (needs the working tree and corpus) "
                     "instead of the released per-speaker tallies")
args = ap.parse_args()

if args.from_preds:
    Q = {(b, s): tally(P / "cdsd_multiseed_preds" / f"s{s}" / f"{b}_s{s}.test.preds.jsonl")
         for s in S for b in ("upper", "all")}
    W = {(b, s): tally(P / "whisper_cdsd_multiseed_preds" / f"s{s}" / f"{b}_s{s}.test.preds.jsonl")
         for s in S for b in ("upper", "all")}
    src = "per-utterance predictions"
else:
    t = from_tallies(P / "cdsd_interaction_speaker_tallies.csv")
    Q = {(b, s): t[("qwen", b, s)] for s in S for b in ("upper", "all")}
    W = {(b, s): t[("whisper", b, s)] for s in S for b in ("upper", "all")}
    src = "released per-speaker tallies"
    missing = [k for k, v in list(Q.items()) + list(W.items()) if not v]
    if missing:
        raise SystemExit(f"tallies file is missing cells: {missing}")

spk = sorted(Q[("upper", 42)])
assert all(sorted(d) == spk for d in list(Q.values()) + list(W.values())), \
    "the two models were not scored on the same speaker set"
print(f"source={src}")
print(f"speakers={spk}  (identical split for both models)\n")


def micro(sel, d):
    e = sum(d[s][0] for s in sel if s in d); n = sum(d[s][1] for s in sel if s in d)
    return 100.0 * e / max(n, 1)


def macro(sel, d):
    return float(np.mean([100.0 * d[s][0] / max(d[s][1], 1) for s in sel if s in d]))


rng = np.random.default_rng(42)
idx = rng.integers(0, len(spk), size=(B, len(spk)))
pq = rng.integers(0, len(S), size=B)
pw = rng.integers(0, len(S), size=B)

for name, fn in (("MICRO", micro), ("MACRO", macro)):
    q = [fn(spk, Q[("upper", s)]) - fn(spk, Q[("all", s)]) for s in S]
    w = [fn(spk, W[("upper", s)]) - fn(spk, W[("all", s)]) for s in S]
    dod = float(np.mean(q) - np.mean(w))
    draws = np.empty(B)
    for i in range(B):
        sel = [spk[j] for j in idx[i]]
        sq, sw = S[pq[i]], S[pw[i]]
        draws[i] = ((fn(sel, Q[("upper", sq)]) - fn(sel, Q[("all", sq)]))
                    - (fn(sel, W[("upper", sw)]) - fn(sel, W[("all", sw)])))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    ok = (lo > 0) == (hi > 0)
    print(f"{name}:  Qwen upper-all = {np.mean(q):+.3f}   Whisper upper-all = {np.mean(w):+.3f}")
    print(f"        INTERACTION (difference of differences) = {dod:+.3f}   "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]   {'RESOLVED' if ok else 'SPANS ZERO'}\n")

# exact sign-flip on the per-speaker interaction
d = []
for k in spk:
    q = np.mean([100.0 * Q[("upper", s)][k][0] / Q[("upper", s)][k][1]
                 - 100.0 * Q[("all", s)][k][0] / Q[("all", s)][k][1] for s in S])
    w = np.mean([100.0 * W[("upper", s)][k][0] / W[("upper", s)][k][1]
                 - 100.0 * W[("all", s)][k][0] / W[("all", s)][k][1] for s in S])
    d.append(q - w)
d = np.array(d)
obs = abs(d.mean())
cnt = sum(1 for sg in itertools.product([1, -1], repeat=len(d))
          if abs((d * np.array(sg)).mean()) >= obs - 1e-12)
print(f"exact paired sign-flip on the interaction: p = {cnt/2**len(d):.4f} "
      f"(floor {2/2**len(d):.4f}, n={len(d)})")
print(f"per-speaker interaction: {[round(x,2) for x in d]}")
