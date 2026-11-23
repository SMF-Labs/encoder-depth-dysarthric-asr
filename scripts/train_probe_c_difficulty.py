#!/usr/bin/env python3
"""Probe C — layer-wise recognition-difficulty prediction from frozen encoder representations.

Muthu et al. §III.D.3 / §IV.B: a linear softmax probe over the mean-pooled hidden state of each encoder
layer predicts a low/medium/high difficulty tier, where the tiers are tertiles of per-utterance CER from
a stronger external decoder. Metric is macro-F1 under 5-fold cross-validation. Their finding is that
difficulty is encoded predominantly in the DEEPEST layers, unlike phoneme identity (mid-to-upper) and
boundary information (weak throughout).

CANDOR deviations, both documented in logs/research_log_2026-08-10.md:
  - Difficulty labels come from `qwen3-asr-1.7b-dys-zh-aug` decoded via qwen_asr (the Track 1 path), not
    Whisper-large. A single fixed label source is used for BOTH the base and zh-aug probes so the two are
    scored against an identical target; the labeller is genuinely external only for the base arm, and the
    zh-aug arm carries a circularity caveat that must be stated.
  - Folds are SPEAKER-disjoint, not random. A random 5-fold over utterances would let the probe memorise
    speaker identity — difficulty is strongly speaker-dependent here (CDSD severity is CER-derived), so
    random folds would inflate macro-F1 substantially.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/train_probe_c_difficulty.py --tag base
    python scripts/layerprobe/train_probe_c_difficulty.py --tag zh_aug
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
FEATS = PROBE_DIR / "features"
TIERS = ["low", "medium", "high"]


def load_pooled(tag: str):
    """Return {"<speaker>/<utt_id>": (25, D)} — keyed by speaker too, ids repeat across shards."""
    pooled, speaker = {}, {}
    for shard in sorted((FEATS / tag).glob("*.npz")):
        with np.load(shard, allow_pickle=True) as z:
            p = z["pooled"].astype(np.float32)
            ids = [str(u) for u in z["utt_ids"]]
        for i, u in enumerate(ids):
            pooled[f"{shard.stem}/{u}"] = p[i]
            speaker[f"{shard.stem}/{u}"] = shard.stem
    return pooled, speaker


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> float:
    fs = []
    for c in range(n_cls):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="base or zh_aug")
    ap.add_argument("--labels", type=Path, default=PROBE_DIR / "cdsd_difficulty_labels.jsonl")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_probe_manifest.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "probeC_difficulty_layerwise.csv")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = {}
    for line in open(args.labels, encoding="utf-8"):
        line = line.strip()
        if line:
            rec = json.loads(line)
            labels[rec["uid"]] = rec["difficulty"]

    # manifest uid is "<speaker>/<subset>/<utt_id>"; feature keys are "<speaker>/<utt_id>"
    feat_key = {}
    for line in open(args.manifest, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            feat_key[r["uid"]] = f"{r['speaker_id']}/{r['utt_id']}"

    pooled, speaker = load_pooled(args.tag)
    keys, y, spk = [], [], []
    for uid, tier in labels.items():
        fk = feat_key.get(uid)
        if fk and fk in pooled:
            keys.append(fk)
            y.append(TIERS.index(tier))
            spk.append(speaker[fk])
    y = np.array(y)
    print(f"{args.tag}: {len(keys)} utterances with both features and labels")
    print(f"  tier distribution: {dict(Counter(TIERS[i] for i in y))}")
    if not keys:
        raise SystemExit("no overlap between features and difficulty labels")

    speakers = sorted(set(spk))
    rng = np.random.default_rng(args.seed)
    order = speakers[:]
    rng.shuffle(order)
    fold_of = {s: i % args.folds for i, s in enumerate(order)}
    spk_arr = np.array([fold_of[s] for s in spk])
    print(f"  speaker-disjoint {args.folds}-fold over {len(speakers)} speakers")

    X_all = np.stack([pooled[k] for k in keys], 0)  # (n, 25, D)
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for layer in range(X_all.shape[1]):
        X = X_all[:, layer, :]
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
        Xn = (X - mu) / sd
        scores = []
        for f in range(args.folds):
            tr, te = spk_arr != f, spk_arr == f
            if te.sum() == 0 or tr.sum() == 0:
                continue
            xt = torch.tensor(Xn[tr], dtype=torch.float32, device=device)
            yt = torch.tensor(y[tr], dtype=torch.long, device=device)
            xv = torch.tensor(Xn[te], dtype=torch.float32, device=device)

            torch.manual_seed(args.seed)
            clf = nn.Linear(X.shape[1], len(TIERS)).to(device)
            opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
            lossf = nn.CrossEntropyLoss()
            for _ in range(args.epochs):
                perm = torch.randperm(xt.shape[0], device=device)
                for i in range(0, len(perm), args.batch_size):
                    idx = perm[i:i + args.batch_size]
                    loss = lossf(clf(xt[idx]), yt[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                pred = clf(xv).argmax(-1).cpu().numpy()
            scores.append(macro_f1(y[te], pred))

        m = float(np.mean(scores))
        results.append({"model": args.tag, "layer": layer, "macro_f1": round(m, 4),
                        "std": round(float(np.std(scores)), 4), "n": len(keys),
                        "folds": len(scores)})
        print(f"layer {layer:2d}: macro-F1 {m:.4f} (±{np.std(scores):.4f})")

        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\nbest: layer {best['layer']} macro-F1 {best['macro_f1']}  "
          f"(chance = {1/len(TIERS):.3f})")
    print(f"wrote {len(results)} rows -> {args.out}")


if __name__ == "__main__":
    main()
