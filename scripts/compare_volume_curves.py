#!/usr/bin/env python3
"""Compare probe PER curves across training volumes, within and across architectures.

The question this answers: Whisper's post-minimum rise collapsed from +6.56 pp (6,200 train) to
+2.07 pp (33,617 train), while Qwen's +1.85 pp was only ever measured at 6,200. If Qwen's rise is
volume-stable, the late-stack shape difference between architectures is an artefact of measuring one
arm at a starved volume. If Qwen's also moves, the quantity is not a stable architectural property.

Usage:
    python scripts/layerprobe/compare_volume_curves.py \
        --qwen-groups "results/probe/qwen_volume_g*.csv" \
        --qwen-base   "results/probe/qwen3asr_zh_probe_layerwise.csv"
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

WHISPER_6200 = {0: 81.79, 1: 71.602, 2: 68.658, 3: 63.804, 4: 63.179, 5: 58.918, 6: 54.249,
                7: 55.004, 8: 51.777, 9: 49.674, 10: 46.003, 11: 49.717, 12: 52.567}
WHISPER_FULL = {0: 76.566, 1: 68.005, 2: 63.313, 3: 60.495, 4: 56.645, 5: 56.058, 6: 51.488,
                7: 50.599, 8: 46.724, 9: 44.963, 10: 41.352, 11: 43.854, 12: 43.421}


def read_group_csvs(pattern: str) -> dict[int, float]:
    curve: dict[int, float] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                curve[int(row["layer"])] = float(row["per_tone_sensitive"])
    return curve


def read_qwen_base(path: Path) -> dict[int, float]:
    curve: dict[int, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("model") == "base" and row.get("metric") == "PER_tone_sensitive":
                curve[int(row["layer"])] = float(row["value"])
    return curve


def shape(curve: dict[int, float]) -> tuple[int, float, float, int]:
    """Return (argmin layer, min PER, rise from min to last layer, last layer)."""
    last = max(curve)
    lo = min(curve, key=lambda k: curve[k])
    return lo, curve[lo], curve[last] - curve[lo], last


def report(name: str, curve: dict[int, float], n_train: int) -> tuple:
    lo, mn, rise, last = shape(curve)
    depth = 100.0 * lo / last
    print(f"{name:<26} n_train={n_train:>6}  min@L{lo:<2} ({depth:4.0f}% depth) = {mn:6.2f}   "
          f"rise L{lo}->L{last} = {rise:+5.2f} pp")
    return lo, mn, rise, depth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-groups", default="results/probe/qwen_volume_g*.csv")
    ap.add_argument("--qwen-base", type=Path,
                    default=Path("results/probe/qwen3asr_zh_probe_layerwise.csv"))
    args = ap.parse_args()

    qf = read_group_csvs(args.qwen_groups)
    missing = [l for l in range(25) if l not in qf]
    if missing:
        print(f"WARNING: qwen full-volume curve missing layers {missing}\n")

    qb = read_qwen_base(args.qwen_base)

    print("=" * 92)
    print("SHAPE SUMMARY")
    print("=" * 92)
    report("Whisper-small @ 6,200", WHISPER_6200, 6200)
    report("Whisper-small @ 33,617", WHISPER_FULL, 33617)
    if qb:
        report("Qwen3-ASR @ 6,200", qb, 6200)
    _, _, qf_rise, _ = report("Qwen3-ASR @ 33,617", qf, 33617)

    print("\n" + "=" * 92)
    print("QWEN PER-LAYER, 6,200 vs 33,617")
    print("=" * 92)
    print(f"{'layer':>5} {'6,200':>9} {'33,617':>9} {'delta':>8}")
    for l in sorted(qf):
        b = qb.get(l)
        d = f"{qf[l]-b:+8.2f}" if b is not None else " " * 8
        bs = f"{b:9.2f}" if b is not None else " " * 9
        print(f"{l:>5} {bs} {qf[l]:9.2f} {d}")

    if qb and not missing:
        w_delta = 2.07 - 6.56
        q_delta = qf_rise - (shape(qb)[2])
        print("\n" + "=" * 92)
        print("VERDICT")
        print("=" * 92)
        print(f"Whisper post-min rise moved {w_delta:+.2f} pp with 5.4x data (6.56 -> 2.07)")
        print(f"Qwen    post-min rise moved {q_delta:+.2f} pp with 5.4x data "
              f"({shape(qb)[2]:.2f} -> {qf_rise:.2f})")
        gap = qf_rise - 2.07
        print(f"\nAt MATCHED high volume: Qwen rise {qf_rise:+.2f} vs Whisper +2.07  ->  gap {gap:+.2f} pp")
        if abs(gap) < 1.0:
            print("=> Late-stack shape difference does NOT survive volume matching. "
                  "Contribution 1 needs a different basis.")
        else:
            print("=> Late-stack shape difference SURVIVES volume matching. "
                  "Contribution 1 can rest on it, reported at matched volume.")


if __name__ == "__main__":
    main()
