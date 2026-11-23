#!/usr/bin/env python3
"""Cross-condition representational divergence across encoder layers.

Muthu et al. §III.C compute, per layer, the cosine similarity between mean-pooled representations of
transcript-matched utterance pairs drawn from three conditions — (d) original dysarthric, (r) speaker-
conditioned TTS resynthesis, (u) unconditioned TTS — and report that divergence grows with depth,
which is their evidence that dysarthria affects high-level representations more than low-level acoustics.

CANDOR cannot form per-utterance triples from the existing XTTS pool (only 3.0% of dys-conditioned rows
pair a transcript with the speaker who actually said it; requiring a matching (u) leaves 80). Per
Bernard's 2026-08-10 decision this runs two analyses side by side:

  distributional  — per-SPEAKER centroids per condition, then cosine between condition centroids.
                    Valid because 93% of the pool's transcripts come from CDSD's own text pool, so the
                    conditions share a lexical distribution and speaker-level pooling averages lexical
                    variation out. Weaker than matched-utterance: it controls lexical content only in
                    expectation, and that must be stated in the writeup.
  strict          — the 411 same-speaker same-transcript (d, r) pairs, as a consistency check. If both
                    give the same direction of divergence-vs-depth, the distributional result stands.

Outputs results/probe/qwen3asr_zh_condition_divergence.csv.

Usage (DGX):
    python scripts/layerprobe/condition_similarity.py --tag-suffix base
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
FEATS = PROBE_DIR / "features"


def load_pooled(tag: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return {"<speaker>/<utt_id>": (25, D) pooled} and {key: speaker} for a feature tag.

    Keyed by speaker AND utt_id: the XTTS pools reuse utterance ids across speakers, so keying on
    utt_id alone silently collapsed 13,627 rows to 2,000.
    """
    pooled: dict[str, np.ndarray] = {}
    speaker: dict[str, str] = {}
    d = FEATS / tag
    if not d.exists():
        raise SystemExit(f"missing features: {d}")
    for shard in sorted(d.glob("*.npz")):
        with np.load(shard, allow_pickle=True) as z:
            p = z["pooled"].astype(np.float32)  # (n, 25, D)
            ids = [str(u) for u in z["utt_ids"]]
        for i, u in enumerate(ids):
            key = f"{shard.stem}/{u}"
            pooled[key] = p[i]
            speaker[key] = shard.stem
    return pooled, speaker


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine over the layer axis: a, b are (25, D)."""
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-8)


def centroids(pooled: dict[str, np.ndarray], speaker: dict[str, str]) -> dict[str, np.ndarray]:
    acc: dict[str, list] = defaultdict(list)
    for u, v in pooled.items():
        acc[speaker[u]].append(v)
    return {s: np.mean(np.stack(v, 0), axis=0) for s, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-suffix", default="base")
    ap.add_argument("--out", type=Path,
                    default=PROBE_DIR / "qwen3asr_zh_condition_divergence.csv")
    args = ap.parse_args()

    d_pool, d_spk = load_pooled(args.tag_suffix)
    r_pool, r_spk = load_pooled(f"cond_r_{args.tag_suffix}")
    u_pool, u_spk = load_pooled(f"cond_u_{args.tag_suffix}")

    # (d) features for the strict-pair check only: the probe set is a 200-utt/speaker subsample, so
    # most paired transcripts have no (d) counterpart extracted. These extras are used ONLY by the
    # strict analysis, never by the distributional one, so they cannot shift the main result.
    extra_dir = FEATS / f"cond_d_extra_{args.tag_suffix}"
    d_strict = dict(d_pool)
    if extra_dir.exists():
        extra, _ = load_pooled(extra_dir.name)
        d_strict.update(extra)
        print(f"strict-pair (d) extras loaded: {len(extra)}")
    print(f"pooled loaded: d={len(d_pool)}  r={len(r_pool)}  u={len(u_pool)}")

    rows = []

    # ---- distributional: global condition centroids, and per-speaker for (d,r) where speakers align
    gc = {
        "d": np.mean(np.stack(list(d_pool.values()), 0), axis=0),
        "r": np.mean(np.stack(list(r_pool.values()), 0), axis=0),
        "u": np.mean(np.stack(list(u_pool.values()), 0), axis=0),
    }
    for pair in (("d", "r"), ("d", "u"), ("r", "u")):
        sim = cos(gc[pair[0]], gc[pair[1]])
        for layer, s in enumerate(sim):
            rows.append({"analysis": "distributional_global", "pair": f"{pair[0]}-{pair[1]}",
                         "layer": layer, "cosine": round(float(s), 5),
                         "distance": round(1 - float(s), 5), "n": len(d_pool)})

    # (d, r) share CDSD speaker identities, so a per-speaker centroid contrast is available and is the
    # tighter of the two distributional views
    dc, rc = centroids(d_pool, d_spk), centroids(r_pool, r_spk)
    shared = sorted(set(dc) & set(rc))
    if shared:
        sims = np.stack([cos(dc[s], rc[s]) for s in shared], 0)  # (n_spk, 25)
        for layer in range(sims.shape[1]):
            col = sims[:, layer]
            rows.append({"analysis": "distributional_per_speaker", "pair": "d-r", "layer": layer,
                         "cosine": round(float(col.mean()), 5),
                         "distance": round(1 - float(col.mean()), 5), "n": len(shared)})
        print(f"per-speaker (d,r) centroids over {len(shared)} shared speakers")

    # ---- strict: 411 same-speaker same-transcript (d, r) pairs
    paired = [json.loads(l) for l in
              open(PROBE_DIR / "cond_paired_manifest.jsonl", encoding="utf-8") if l.strip()]
    text_of_d = {}
    for name in ("cond_d_manifest.jsonl", "cond_d_extra_manifest.jsonl"):
        path = PROBE_DIR / name
        if not path.exists():
            continue
        for rec in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()):
            text_of_d.setdefault(
                (rec["speaker_id"], rec["text"]), f"{rec['speaker_id']}/{rec['utt_id']}"
            )

    hits = []
    for rec in paired:
        d_key = text_of_d.get((rec["speaker_id"], rec["text"]))
        r_key = f"{rec['speaker_id']}/{rec['utt_id']}"
        if d_key and d_key in d_strict and r_key in r_pool:
            hits.append((d_strict[d_key], r_pool[r_key]))
    if hits:
        sims = np.stack([cos(a, b) for a, b in hits], 0)
        for layer in range(sims.shape[1]):
            col = sims[:, layer]
            rows.append({"analysis": "strict_paired", "pair": "d-r", "layer": layer,
                         "cosine": round(float(col.mean()), 5),
                         "distance": round(1 - float(col.mean()), 5), "n": len(hits)})
    print(f"strict paired (d,r) utterance pairs usable: {len(hits)} of {len(paired)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["analysis", "pair", "layer", "cosine", "distance", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

    # ---- readable summary: does divergence grow with depth?
    print("\ncosine distance (1 - cos) by depth:")
    hdr = f"{'analysis':28s} {'pair':5s}" + "".join(f"{f'L{l}':>8s}" for l in (0, 6, 12, 18, 24))
    print(hdr)
    for key in sorted({(r["analysis"], r["pair"]) for r in rows}):
        sel = {r["layer"]: r["distance"] for r in rows
               if (r["analysis"], r["pair"]) == key}
        line = f"{key[0]:28s} {key[1]:5s}" + "".join(f"{sel.get(l, float('nan')):8.4f}" for l in (0, 6, 12, 18, 24))
        trend = "RISES with depth" if sel.get(24, 0) > sel.get(0, 0) else "falls with depth"
        print(f"{line}   {trend}")


if __name__ == "__main__":
    main()
