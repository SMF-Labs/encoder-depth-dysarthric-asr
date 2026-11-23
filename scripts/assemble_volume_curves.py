#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the two matched-volume probe curves Figure 3 needs from their per-layer / per-group shards.

The matched-volume sweeps ran one process per layer group -- see Section 10's note on unified-memory
hardware, where a single process sweeping all layers is killed at the first layer transition -- so these
curves only ever existed as shards. make_figures.py expects them aggregated.

The assembled files are committed, so this script is not needed to reproduce the paper; it documents and
reproduces how they were built from the raw sweep output.

Usage:
    python scripts/assemble_volume_curves.py --whisper-shards <dir> --qwen-shards <dir>
    python scripts/assemble_volume_curves.py --whisper-shards <dir> --qwen-shards <dir> --out results/

Shard filenames are matched as volume_layer_*.csv (Whisper-small, one file per layer) and
qwen_volume_g*.csv (Qwen3-ASR, one file per layer group). Each shard carries the standard probe columns.
"""
import argparse
import csv
import glob
import io
import os

COLS = ["model", "layer", "per_tone_sensitive", "per_tone_agnostic", "val_per", "n_train", "n_test"]

OUTPUTS = [
    ("whisper_volume_probe_layerwise.csv", "whisper_shards", "volume_layer_*.csv"),
    ("qwen_volume_probe_layerwise.csv", "qwen_shards", "qwen_volume_g*.csv"),
]


def assemble(shard_paths, label):
    """Merge shards into one layer-indexed curve, refusing gaps, conflicts and mixed conditions."""
    rows = {}
    for p in shard_paths:
        with io.open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                layer = int(r["layer"])
                if layer in rows and rows[layer] != r:
                    raise SystemExit(f"{label}: conflicting rows for layer {layer} (last seen in {p})")
                rows[layer] = r

    if not rows:
        raise SystemExit(f"{label}: no shards matched")

    keys = sorted(rows)
    if keys != list(range(min(keys), max(keys) + 1)):
        raise SystemExit(f"{label}: gap in layer coverage: {keys}")

    models = {rows[k]["model"] for k in keys}
    volumes = {rows[k]["n_train"] for k in keys}
    if len(models) != 1 or len(volumes) != 1:
        raise SystemExit(f"{label}: shards mix conditions -- models={models}, n_train={volumes}")

    return keys, rows, models.pop(), volumes.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper-shards", required=True,
                    help="directory holding the Whisper-small per-layer shards (volume_layer_*.csv)")
    ap.add_argument("--qwen-shards", required=True,
                    help="directory holding the Qwen3-ASR per-group shards (qwen_volume_g*.csv)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results"),
        help="output directory (default: this repository's results/)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for out_name, arg_name, pattern in OUTPUTS:
        shard_dir = getattr(args, arg_name)
        shards = sorted(glob.glob(os.path.join(shard_dir, pattern)))
        keys, rows, model, volume = assemble(shards, out_name)

        dest = os.path.join(args.out, out_name)
        with io.open(dest, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
            w.writeheader()
            for k in keys:
                w.writerow({c: rows[k][c] for c in COLS})

        vals = [float(rows[k]["per_tone_sensitive"]) for k in keys]
        lo = min(vals)
        print(f"{out_name}: {len(shards)} shards -> layers {keys[0]}-{keys[-1]}, "
              f"model={model}, n_train={volume}")
        print(f"    min PER {lo:.2f} at layer {keys[vals.index(lo)]}, final {vals[-1]:.2f}, "
              f"rise from min +{vals[-1] - lo:.2f} pp")


if __name__ == "__main__":
    main()
