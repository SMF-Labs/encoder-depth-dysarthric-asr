#!/usr/bin/env python3
"""Build manifests for the three-condition similarity arm.

Muthu et al. compare, per encoder layer, the representations of three transcript-matched conditions:
  (d) original dysarthric speech
  (r) zero-shot TTS resynthesis conditioned on the dysarthric speaker
  (u) unconditioned / neutral-speaker TTS

CANDOR's existing `wav_xtts_zh` pool cannot supply per-utterance triples: of its 13,627 CDSD-conditioned
rows, only 3.0% (411) pair a transcript with the speaker who actually said it, and requiring a matching
(u) of the same text leaves 80. Per Bernard's 2026-08-10 decision we therefore run a **matched-
distribution** design — 93% of the pool's transcripts are drawn from CDSD's own text pool, so the three
conditions share a lexical distribution and per-speaker mean pooling averages lexical variation out —
plus the 411 strictly-paired rows as an exact-match consistency check.

This writes three manifests:
    cond_d_manifest.jsonl        CDSD originals (the probe set)
    cond_r_manifest.jsonl        XTTS, ref_kind=dys,  ref_dataset=CDSD
    cond_u_manifest.jsonl        XTTS, ref_kind=hc
    cond_paired_manifest.jsonl   the strict subset where (r) text == a text its own speaker said

Usage (DGX):
    python scripts/layerprobe/build_condition_manifests.py
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
CDSD = BASE / "data" / "processed" / "CDSD" / "manifest_v2.jsonl"
XTTS = BASE / "data" / "processed" / "wav_xtts_zh" / "manifest.jsonl"
OUT = BASE / "results" / "probe"


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write(rows: list[dict], path: Path, label: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    spk = len({r["speaker_id"] for r in rows})
    print(f"  {label:26s} {len(rows):6d} utts / {spk:3d} speakers -> {path.name}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # texts each CDSD speaker actually produced, for the strict-pair check
    spoken: dict[str, set] = defaultdict(set)
    for rec in read_jsonl(CDSD):
        spoken[rec["speaker_id"]].add(rec["text"])

    cond_r, cond_u, paired = [], [], []
    seen: Counter = Counter()
    for rec in read_jsonl(XTTS):
        path = rec["audio_filepath"]
        if not os.path.isabs(path):
            path = str(BASE / path)
        if not os.path.exists(path):
            seen["audio_missing"] += 1
            continue
        spk = rec.get("speaker_id") or "unknown"
        base = {
            "utt_id": Path(rec["audio_filepath"]).stem,
            "uid": f"{spk}/{Path(rec['audio_filepath']).stem}",
            "audio_filepath": path,
            "speaker_id": spk,
            "text": rec.get("text", ""),
            "duration": rec.get("duration"),
            "ref_dataset": rec.get("ref_dataset"),
            "ref_kind": rec.get("ref_kind"),
            "split": "all",
        }
        if rec.get("ref_kind") == "dys" and rec.get("ref_dataset") == "CDSD":
            cond_r.append(base)
            if rec.get("text") in spoken.get(spk, ()):
                paired.append(base)
        elif rec.get("ref_kind") == "hc":
            cond_u.append(base)

    probe_rows = list(read_jsonl(OUT / "cdsd_probe_manifest.jsonl"))

    print("condition manifests:")
    write(probe_rows, OUT / "cond_d_manifest.jsonl", "(d) CDSD original")
    write(cond_r, OUT / "cond_r_manifest.jsonl", "(r) XTTS dys-conditioned")
    write(cond_u, OUT / "cond_u_manifest.jsonl", "(u) XTTS hc-conditioned")
    write(paired, OUT / "cond_paired_manifest.jsonl", "(r) strict same-speaker")

    if seen:
        print(f"  skipped: {dict(seen)}")
    print(
        f"\nstrict-pair rate: {len(paired)}/{len(cond_r)} "
        f"({100*len(paired)/max(len(cond_r),1):.1f}%) — the reason the design is "
        f"matched-distribution rather than matched-utterance"
    )


if __name__ == "__main__":
    main()
