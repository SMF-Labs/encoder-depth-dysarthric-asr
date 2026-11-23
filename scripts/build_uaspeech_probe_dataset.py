#!/usr/bin/env python3
"""Build a UASpeech phoneme-probe manifest, so the depth probe can be run on the second corpus.

WHY THIS DOES NOT USE FORCED ALIGNMENT, and what that costs.

The CDSD probe took its phone inventory from Montreal Forced Aligner TextGrids. That route is closed
for UASpeech: MFA is not installed on this machine, and UASpeech is not redistributable, so the audio
cannot be sent to the cloud instance where MFA lives. Instead the targets come from a pronunciation
dictionary (CMUdict), which is viable here for two specific reasons:

  * The probe is CTC. It needs the target phone SEQUENCE, not time-aligned boundaries.
  * UASpeech is a closed 449-word isolated-word vocabulary, so a dictionary covers it almost entirely
    and there is no out-of-vocabulary tail to guess at.

Two honest consequences, both of which belong in the paper rather than in a footnote:

  1. The targets are CANONICAL pronunciations, not transcriptions of what the speaker actually produced.
     For dysarthric speech that gap is real. It is, however, narrower than it first appears: forced
     alignment also imposes the canonical phone string and only chooses where to place its boundaries,
     and CDSD's tone units were themselves transcript-derived (see build_probe_dataset.py).
  2. There is NO integrity gate. CDSD dropped 14.6% of utterances whose alignment produced no real
     phones — a screen that also removed unusable audio. Nothing equivalent exists here, so the
     UASpeech probe sees the full split. Any comparison of ABSOLUTE PER between the two corpora is
     therefore unsafe; only the shape across depth is comparable.

Words absent from CMUdict are recovered by compound decomposition (backspace -> back + space) where a
split into two dictionary words exists; anything still missing is dropped and reported. Stress digits
are stripped, so ARPAbet phones are the unit.

Usage:
    python scripts/layerprobe/build_uaspeech_probe_dataset.py
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROBE = BASE / "results" / "probe"


def load_cmudict(path: Path) -> dict[str, list[str]]:
    d: dict[str, list[str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
            if word not in d:                      # first listed pronunciation
                d[word] = [re.sub(r"\d", "", p) for p in parts[1:]]
    return d


def pronounce(word: str, d: dict[str, list[str]]) -> list[str] | None:
    w = word.strip().lower()
    if w in d:
        return d[w]
    # hyphenated compounds: able-bodied -> able + bodied
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        if all(p in d for p in parts):
            out: list[str] = []
            for p in parts:
                out += d[p]
            return out
    # solid compounds: backspace -> back + space. Prefer the most balanced split so that
    # "chambermaid" becomes chamber+maid rather than cham+bermaid.
    best = None
    for i in range(3, len(w) - 2):
        a, b = w[:i], w[i:]
        if a in d and b in d:
            score = abs(len(a) - len(b))
            if best is None or score < best[0]:
                best = (score, d[a] + d[b])
    return best[1] if best else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROBE / "uaspeech_lora_manifest.jsonl")
    ap.add_argument("--cmudict", type=Path, default=PROBE / "cmudict.dict")
    ap.add_argument("--out", type=Path, default=PROBE / "uaspeech_probe_manifest.jsonl")
    args = ap.parse_args()

    d = load_cmudict(args.cmudict)
    print(f"cmudict: {len(d)} words")

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    print(f"source manifest: {len(rows)} utterances")

    direct = compound = dropped = 0
    missing_words: Counter = Counter()
    out_rows = []
    phones: Counter = Counter()
    for r in rows:
        w = r["text"].strip().lower()
        if w in d:
            units = d[w]
            direct += 1
        else:
            units = pronounce(w, d)
            if units is None:
                dropped += 1
                missing_words[w] += 1
                continue
            compound += 1
        phones.update(units)
        out_rows.append({**r, "units": units, "n_units": len(units),
                         "unit_source": "cmudict_direct" if w in d else "cmudict_compound"})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    kept = len(out_rows)
    print(f"\nkept {kept}/{len(rows)} utterances ({100*kept/len(rows):.1f}%)")
    print(f"  direct dictionary hit : {direct}")
    print(f"  compound decomposition: {compound}")
    print(f"  dropped (no pronunciation): {dropped} across {len(missing_words)} word types")
    if missing_words:
        print("  dropped word types:", ", ".join(sorted(missing_words)[:20]))
    print(f"\nphone inventory: {len(phones)} distinct ARPAbet phones")
    by_split: Counter = Counter(r["split"] for r in out_rows)
    print("split sizes:", dict(by_split))
    spk = {s: {r["speaker_id"] for r in out_rows if r["split"] == s} for s in by_split}
    print("speakers per split:", {k: len(v) for k, v in spk.items()})
    for a in by_split:
        for b in by_split:
            if a < b and spk[a] & spk[b]:
                raise SystemExit(f"SPEAKER LEAK between {a} and {b}: {spk[a] & spk[b]}")
    print("speaker-disjoint: OK")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
