#!/usr/bin/env python3
"""Two-dimensional speaker x seed uncertainty for the UASpeech band sweep.

Why. The speaker bootstrap in bootstrap_band_cis.py conditions on ONE set of learned weights: it
answers "how much would this contrast move if we had drawn different speakers", not "how much would it
move if we retrained". With three matched seeds per band we can now resample BOTH, which is what the
external review asked for.

Three estimates are reported per contrast, and they answer different questions:
  seed-only     : spread of the paired contrast across seeds, speakers held fixed (all 5).
  speaker-only  : paired speaker bootstrap at a single seed (comparable to the earlier analysis).
  speaker x seed: resample speakers AND seeds together -- the honest interval.

Contrasts are always computed WITHIN a seed before being combined, so a seed's overall difficulty
cancels rather than inflating the contrast.

Usage:
    python scripts/layerprobe/seed_speaker_analysis.py --b 10000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
PREDS = BASE / "results" / "probe" / "seedsweep" / "preds"
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")

BANDS = ["upper", "middle", "all", "lower"]
PAIRS = [("upper", "middle"), ("upper", "all"), ("all", "middle"),
         ("upper", "lower"), ("lower", "middle")]


def norm(t: str) -> str:
    # UASpeech convention: case-folded, punctuation-stripped (see eval_lora_variants.py).
    return _PUNCT.sub("", (t or "").lower())


def _edit(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load() -> tuple[list[str], list[int], dict]:
    """-> speakers, seeds, {(band, seed): {speaker: (err, ref)}}"""
    data: dict = {}
    speakers, seeds = set(), set()
    for p in sorted(PREDS.glob("*.test.preds.jsonl")):
        m = re.match(r"(\w+?)_s(\d+)\.test\.preds\.jsonl$", p.name)
        if not m:
            continue
        band, seed = m.group(1), int(m.group(2))
        acc = defaultdict(lambda: [0, 0])
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            ref, hyp = list(norm(r["ref"])), list(norm(r["hyp"]))
            acc[r["speaker_id"]][0] += _edit(ref, hyp)
            acc[r["speaker_id"]][1] += len(ref)
            speakers.add(r["speaker_id"])
        data[(band, seed)] = {s: tuple(v) for s, v in acc.items()}
        seeds.add(seed)
    return sorted(speakers), sorted(seeds), data


def cer(sel: list[str], d: dict) -> float:
    e = sum(d[s][0] for s in sel if s in d)
    n = sum(d[s][1] for s in sel if s in d)
    return 100.0 * e / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=BASE / "results" / "probe" / "seed_speaker_analysis.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    speakers, seeds, data = load()
    print(f"speakers: {len(speakers)} {speakers}")
    print(f"seeds   : {seeds}")
    missing = [(b, s) for b in BANDS for s in seeds if (b, s) not in data]
    if missing:
        print(f"MISSING runs: {missing}")

    print(f"\n{'band':8s} " + " ".join(f"s{s:<7d}" for s in seeds) + "   mean     SD    range")
    per_band = {}
    for b in BANDS:
        vals = [cer(speakers, data[(b, s)]) for s in seeds if (b, s) in data]
        per_band[b] = vals
        print(f"{b:8s} " + " ".join(f"{v:7.3f} " for v in vals) +
              f"  {np.mean(vals):7.3f} {np.std(vals, ddof=1):6.3f} {max(vals)-min(vals):7.3f}")

    rows = []
    print(f"\n{'contrast':18s} {'mean':>7s}  {'seed-only':>18s}  {'speaker x seed 95% CI':>24s}  sign")
    for a, b in PAIRS:
        # Paired within seed: contrast computed per seed, then combined.
        per_seed = [cer(speakers, data[(a, s)]) - cer(speakers, data[(b, s)])
                    for s in seeds if (a, s) in data and (b, s) in data]
        mean_d = float(np.mean(per_seed))
        sd_d = float(np.std(per_seed, ddof=1))

        # speaker x seed: resample both dimensions.
        draws = np.empty(args.b)
        for i in range(args.b):
            sel = [speakers[j] for j in rng.integers(0, len(speakers), len(speakers))]
            ss = [seeds[j] for j in rng.integers(0, len(seeds), len(seeds))]
            draws[i] = np.mean([cer(sel, data[(a, s)]) - cer(sel, data[(b, s)]) for s in ss])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        sign = "stable" if all(np.sign(x) == np.sign(per_seed[0]) for x in per_seed) else "FLIPS"
        excl = "excludes 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"{a+' - '+b:18s} {mean_d:7.3f}  [{min(per_seed):7.3f},{max(per_seed):7.3f}]  "
              f"[{lo:8.3f}, {hi:8.3f}] {excl:10s} {sign}")
        rows.append({"contrast": f"{a}-{b}", "mean": round(mean_d, 3),
                     "seed_sd": round(sd_d, 3), "seed_min": round(min(per_seed), 3),
                     "seed_max": round(max(per_seed), 3), "lo": round(lo, 3),
                     "hi": round(hi, 3), "excludes_zero": excl == "excludes 0",
                     "sign_stable_across_seeds": sign == "stable"})

    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
