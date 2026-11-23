#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact paired sign-flip (randomisation) tests for the headline contrasts.

WHY. Every interval in this paper comes from a percentile cluster bootstrap over 6 CDSD or 5 UASpeech
speakers. At that few clusters, percentile tail coverage is not guaranteed. With n <= 6 the paired
randomisation distribution is small enough to enumerate EXACTLY -- 2^6 = 64 sign patterns on CDSD, 2^5 =
32 on UASpeech -- so the conclusions can be checked against an estimator that makes no distributional
assumption at all.

The statistic is the mean per-speaker CER difference; under the null of exchangeable signs the sign of
each speaker's difference is flipped over all 2^n assignments. Reported two-sided.

This does not replace the bootstrap; it is a robustness check on it. The smallest attainable two-sided
p-value is 2/2^n -- 0.031 at n=6, 0.062 at n=5 -- so a contrast can be unanimous across speakers and
still not reach 0.05 on UASpeech. That floor is itself worth reporting.
"""
import itertools, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

PROBE = Path.home() / "dysarthria" / "results" / "probe"
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


def per_speaker_cer(path, case_fold=False, word=False):
    acc = defaultdict(lambda: [0, 0])
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        rt, ht = r["ref"] or "", r["hyp"] or ""
        if case_fold:
            rt, ht = rt.lower(), ht.lower()
        ref = rt.split() if word else list(_P.sub("", rt))
        hyp = ht.split() if word else list(_P.sub("", ht))
        acc[r["speaker_id"]][0] += _edit(ref, hyp)
        acc[r["speaker_id"]][1] += len(ref)
    return {s: 100.0 * e / max(n, 1) for s, (e, n) in acc.items()}


def signflip(diffs):
    """Exact two-sided paired randomisation p-value over all 2^n sign assignments."""
    d = np.array(diffs, dtype=float)
    n = len(d)
    obs = abs(d.mean())
    count = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((d * np.array(signs)).mean()) >= obs - 1e-12:
            count += 1
    return count / (2 ** n), 2.0 / (2 ** n)


def run(label, a_paths, b_paths, case_fold=False, word=False):
    """a_paths/b_paths are per-seed lists; the contrast is averaged over seeds within speaker."""
    A = [per_speaker_cer(p, case_fold, word) for p in a_paths]
    B = [per_speaker_cer(p, case_fold, word) for p in b_paths]
    spk = sorted(A[0])
    diffs = [float(np.mean([A[i][s] - B[i][s] for i in range(len(A))])) for s in spk]
    p, floor = signflip(diffs)
    star = "SIG" if p < 0.05 else "n.s."
    print(f"{label:34s} mean {np.mean(diffs):+7.3f}  exact p = {p:.4f} "
          f"(floor {floor:.4f}, n={len(spk)})  {star}")
    print(f"{'':34s} per-speaker {[round(x, 2) for x in diffs]}")


print("=== CDSD x Qwen3-ASR (3 seeds) ===")
ms = PROBE / "cdsd_multiseed_preds"
S = [42, 43, 44]
for a, b in (("upper", "all"), ("upper", "middle"), ("upper", "lower"), ("middle", "all")):
    run(f"{a} - {b}",
        [ms / f"s{s}" / f"{a}_s{s}.test.preds.jsonl" for s in S],
        [ms / f"s{s}" / f"{b}_s{s}.test.preds.jsonl" for s in S])

print("\n=== CDSD x Qwen3-ASR, alpha-matched 2x2 (3 seeds) ===")
am = PROBE / "alpha_matched_preds"
for a, b in (("upper_r192", "all_r64"), ("upper_r64", "all_r21")):
    run(f"{a} - {b}",
        [am / f"s{s}" / f"{a}_s{s}.test.preds.jsonl" for s in S],
        [am / f"s{s}" / f"{b}_s{s}.test.preds.jsonl" for s in S])

print("\n=== UASpeech x Whisper-small (3 seeds, WER) ===")
uw = PROBE / "whisper_uaspeech_preds_fixed"
for a, b in (("upper", "all"), ("upper", "middle")):
    run(f"{a} - {b} (WER)",
        [uw / f"s{s}" / f"{a}_s{s}.test.preds.jsonl" for s in S],
        [uw / f"s{s}" / f"{b}_s{s}.test.preds.jsonl" for s in S],
        case_fold=True, word=True)
