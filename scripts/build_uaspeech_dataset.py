#!/usr/bin/env python3
"""Build the speaker-disjoint UASpeech dataset for the second-corpus replication of the band sweep.

Why UASpeech. The Mandarin study's binding limitations are (a) one language and (b) six test speakers.
UASpeech addresses the first directly and gives a *different content type* — 449 isolated words rather
than read sentences — so it tests whether the depth ordering depends on continuous speech. It also
carries clinical mild/severe labels, where CDSD's severities are CER-derived pseudo-labels.

What it does NOT fix: UASpeech has only 15 dysarthric speakers, so a stratified split yields 5 test
speakers, comparable to CDSD's 6 rather than better. Every speaker records the same 449-word list, so a
text-disjoint split is impossible by construction — the task is closed-vocabulary by design. Splits are
speaker-disjoint only, and that must be stated wherever these numbers appear.

Split is stratified by severity and seeded, not hand-picked:
    mild   (5 speakers) -> 2 test / 1 val / 2 train
    severe (10 speakers) -> 3 test / 1 val / 6 train

Usage (DGX):
    python scripts/layerprobe/build_uaspeech_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
MANIFEST = BASE / "data" / "processed" / "UASPEECH" / "manifest_v2.jsonl"
OUT_DIR = BASE / "results" / "probe"

EXPECTED_SPEAKERS = 15

# Stratify on severity AND gender, not severity alone. A severity-only split put all four female
# speakers in training and produced an all-male test set — the exact demographic-coverage weakness
# that produced the CDSD_08 failure in the Mandarin study, and not one to reproduce by accident.
# The single mild-female speaker is assigned to train so that training covers all four cells;
# with n=1 she cannot be in both, and leaving a training cell empty is the worse option.
# cell -> (n_test, n_val, n_train); cells are (severity, gender)
CELL_ALLOC = {
    ("severe", "female"): (1, 0, 2),   # 3 speakers
    ("severe", "male"): (2, 1, 4),     # 7 speakers
    ("mild", "female"): (0, 0, 1),     # 1 speaker
    ("mild", "male"): (2, 1, 1),       # 4 speakers
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dur", type=float, default=0.2)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "uaspeech_lora_manifest.jsonl")
    args = ap.parse_args()

    rows = []
    reasons = Counter()
    total = 0
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            r = json.loads(line)
            dur = float(r.get("duration") or 0.0)
            if not (args.min_dur <= dur <= args.max_dur):
                reasons["duration_out_of_range"] += 1
                continue
            # UASpeech manifest paths are relative to the CORPUS directory ("train/F02_B1_UW51.wav"),
            # not to DYSARTHRIA_BASE as in most other processed corpora.
            path = r["audio_filepath"]
            if not os.path.isabs(path):
                path = str(MANIFEST.parent / path)
            if not os.path.exists(path):
                reasons["audio_missing"] += 1
                continue
            if not (r.get("text") or "").strip():
                reasons["empty_text"] += 1
                continue
            rows.append({
                "uid": f"{r['speaker_id']}/{Path(r['audio_filepath']).stem}",
                "utt_id": Path(r["audio_filepath"]).stem,
                "speaker_id": r["speaker_id"],
                "audio_filepath": path,
                "text": r["text"],
                "duration": dur,
                "pseudo_severity": r.get("severity"),
                "severity_source": "clinical (UASpeech intelligibility tier)",
                "gender": "female" if r["speaker_id"].startswith("F") else "male",
                "subset": "uaspeech",
            })

    by_cell: dict[tuple, set] = defaultdict(set)
    for r in rows:
        by_cell[(r["pseudo_severity"], r["gender"])].add(r["speaker_id"])
    speakers = {s for v in by_cell.values() for s in v}
    if len(speakers) != EXPECTED_SPEAKERS:
        raise SystemExit(f"compose-check: {len(speakers)} speakers, expected {EXPECTED_SPEAKERS}")

    split_of: dict[str, str] = {}
    rng = random.Random(args.seed)
    for cell in sorted(by_cell):
        pool = sorted(by_cell[cell])
        rng.shuffle(pool)
        n_te, n_va, n_tr = CELL_ALLOC.get(cell, (max(1, len(pool) // 5), 1, 0))
        if n_te + n_va + n_tr != len(pool):
            raise SystemExit(
                f"CELL_ALLOC for {cell} sums to {n_te+n_va+n_tr} but the cell holds {len(pool)} "
                f"speakers ({pool}) — the corpus composition has changed"
            )
        for i, s in enumerate(pool):
            split_of[s] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")

    for r in rows:
        r["split"] = split_of[r["speaker_id"]]

    # speaker-leak gate
    spk_by_split = defaultdict(set)
    for r in rows:
        spk_by_split[r["split"]].add(r["speaker_id"])
    leaks = {}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and (spk_by_split[a] & spk_by_split[b]):
                leaks[f"{a}|{b}"] = sorted(spk_by_split[a] & spk_by_split[b])
    if leaks:
        raise SystemExit(f"SPEAKER LEAK: {leaks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in sorted(rows, key=lambda r: r["uid"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"source utts: {total}   filtered: {dict(reasons) or 'none'}")
    print(f"written    : {len(rows)} -> {args.out}")
    print(f"\n{'split':6s}{'spk':>5s}{'utts':>7s}  severity   speakers")
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        cells = Counter((r["pseudo_severity"], r["gender"]) for r in sub)
        pretty = {f"{k[0]}/{k[1][0]}": v for k, v in sorted(cells.items())}
        print(f"{s:6s}{len(spk_by_split[s]):5d}{len(sub):7d}  {pretty}  {sorted(spk_by_split[s])}")
    print(f"\nspeaker overlap across splits: {len(leaks)} (must be 0)")
    texts = {r["text"] for r in rows}
    print(f"unique texts: {len(texts)} (closed vocabulary — splits are speaker-disjoint only)")
    uids = Counter(r["uid"] for r in rows)
    print(f"uid uniqueness: {len(uids)}/{len(rows)}  duplicates={sum(1 for v in uids.values() if v > 1)}")


if __name__ == "__main__":
    main()
