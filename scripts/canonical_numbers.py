#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CANONICAL NUMBERS for Paper 10a. One source of truth for every result-bearing value.

Written after a reviewer identified the paper's actual failure mode: not wrong results, but PROPAGATION
- a three-seed rewrite corrected the tables while a dozen sentences kept speaking the superseded
single-seed values, and a regret table was built from a single-run CSV while the paper reported
three-seed means for the same cell.

Spot-fixing that class of error guarantees the next reader finds the cell nobody flagged. Every
result-bearing table, figure and sentence should be regenerated from THIS bundle instead.

Emits JSON to results/probe/canonical_numbers.json and a human-readable summary to stdout.
"""
import csv, glob, json, os, statistics as st
import os
from pathlib import Path

# Results directory: this repository's own results/ by default, so the script runs from a clean
# clone. Set P10A_RESULTS to point at a working tree instead.
P = Path(os.environ.get("P10A_RESULTS",
                        Path(__file__).resolve().parent.parent / "results"))
SEEDS = [42, 43, 44]
BANDS = ["lower", "middle", "upper", "all"]


def mean_sd(v):
    return {"mean": round(st.mean(v), 3),
            "sd": round(st.stdev(v), 3) if len(v) > 1 else 0.0,
            "per_seed": [round(x, 3) for x in v], "n_seeds": len(v)}


def from_multiseed(pattern, key="cer", strip=True):
    """Per-band values from one CSV per seed."""
    acc = {}
    for f in sorted(P.glob(pattern)):
        for r in csv.DictReader(open(f)):
            v = r["variant"]
            if strip:
                for s in SEEDS:
                    v = v.replace(f"_s{s}", "")
            try:
                acc.setdefault(v, []).append(float(r[key]))
            except (TypeError, ValueError):
                pass
    return acc


def from_per_band(prefix, key="cer"):
    """Per-band values from one CSV per (band, seed), e.g. seedsweep/upper_s42.csv."""
    acc = {}
    for b in BANDS:
        vals = []
        for s in SEEDS:
            f = P / f"{prefix}/{b}_s{s}.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                if r.get("variant", b).replace(f"_s{s}", "") in (b, "adapted", ""):
                    try:
                        vals.append(float(r[key])); break
                    except (TypeError, ValueError):
                        pass
        if vals:
            acc[b] = vals
    return acc


C = {}

# --- Qwen3-ASR x CDSD, three seeds -------------------------------------------------
a = from_multiseed("cdsd_multiseed_s*.csv")
C["qwen_cdsd"] = {"metric": "CER", "seeds": SEEDS,
                  "bands": {b: mean_sd(a[b]) for b in BANDS if b in a},
                  "base": round(st.mean(a["base"]), 3) if "base" in a else None}

# --- Whisper-small x CDSD, three seeds ---------------------------------------------
a = from_multiseed("whisper_cdsd_multiseed_s*.csv")
C["whisper_cdsd"] = {"metric": "CER", "seeds": SEEDS,
                     "bands": {b: mean_sd(a[b]) for b in BANDS if b in a},
                     "base": round(st.mean(a["base"]), 3) if "base" in a else None}

# --- Whisper-small x UASpeech, three seeds (corrected decode) ----------------------
for k, metric in (("cer", "CER"), ("wer", "WER")):
    a = from_multiseed("whisper_uaspeech_sweep_fixed_s*.csv", key=k)
    C[f"whisper_uaspeech_{k}"] = {"metric": metric, "seeds": SEEDS,
                                  "bands": {b: mean_sd(a[b]) for b in BANDS if b in a},
                                  "base": round(st.mean(a["base"]), 3) if "base" in a else None}

# --- Qwen3-ASR x UASpeech, three seeds --------------------------------------------
a = from_per_band("seedsweep")
if a:
    C["qwen_uaspeech"] = {"metric": "CER", "seeds": SEEDS,
                          "bands": {b: mean_sd(a[b]) for b in BANDS if b in a},
                          "base": None, "note": "base from single-run target sweep"}
    for r in csv.DictReader(open(P / "qwen3asr_uaspeech_target_sweep.csv")):
        if r["variant"] == "base":
            C["qwen_uaspeech"]["base"] = float(r["cer"])

# --- alpha-matched 2x2 -------------------------------------------------------------
a = from_multiseed("alpha_matched_s*.csv")
C["alpha_matched"] = {"metric": "CER", "seeds": SEEDS,
                      "cells": {k: mean_sd(v) for k, v in a.items() if k != "base"},
                      "base": round(st.mean(a["base"]), 3) if "base" in a else None}

out = P / "canonical_numbers.json"
json.dump(C, open(out, "w"), indent=2)

print("CANONICAL NUMBERS  (source of truth; regenerate the manuscript from these)\n")
for cell in ("qwen_cdsd", "whisper_cdsd", "qwen_uaspeech", "whisper_uaspeech_cer", "whisper_uaspeech_wer"):
    if cell not in C:
        continue
    d = C[cell]
    print(f"== {cell}  ({d['metric']}, {len(d['seeds'])} seeds)   base = {d['base']}")
    ranked = sorted(d["bands"].items(), key=lambda kv: kv[1]["mean"])
    for b, v in ranked:
        print(f"   {b:7s} {v['mean']:7.3f} +/- {v['sd']:.3f}   {v['per_seed']}")
    best = ranked[0][0]
    print(f"   best = {best};  regret: " + "  ".join(
        f"{b}:+{v['mean']-ranked[0][1]['mean']:.2f}" for b, v in ranked[1:]))
    print()
print("== alpha-matched 2x2   base =", C["alpha_matched"]["base"])
for k, v in sorted(C["alpha_matched"]["cells"].items()):
    print(f"   {k:12s} {v['mean']:7.3f} +/- {v['sd']:.3f}")
print(f"\nwrote {out}")
