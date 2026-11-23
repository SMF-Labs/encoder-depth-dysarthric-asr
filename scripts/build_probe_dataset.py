#!/usr/bin/env python3
"""Build the speaker-disjoint CDSD probe dataset for the layer-wise probing study.

Follows Muthu et al. (arXiv:2608.01865) where the design carries over:
  - speaker-disjoint 70/15/15 train/val/test split
  - phoneme labels as pinyin initial / final / tone units (tong2 -> t, ong, T2)

Deviations, both forced and documented in logs/research_log_2026-08-10.md:
  - Probe A (phoneme boundary detection) is dropped: the Qwen3-ASR encoder runs at ~77 ms/frame
    against a median CDSD phone of 80 ms, so 49% of phones are sub-frame and boundary labels are
    not well posed. The MFA alignments are therefore used only to screen unusable utterances.
  - Tone comes from the transcript via pypinyin, NOT from the TextGrid tone letters: T1 is absent
    from all 43,137 aligned phone tokens, so the alignments cannot support a 4-way tone probe.

Outputs results/probe/cdsd_probe_manifest.jsonl with, per utterance, the tone-sensitive and
tone-agnostic unit sequences that Probe B scores PER against.

Usage (DGX):
    python scripts/layerprobe/build_probe_dataset.py --per-speaker 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
MANIFEST = BASE / "data" / "processed" / "CDSD" / "manifest_v2.jsonl"
ALIGNED = BASE / "results" / "track4" / "aligned" / "CDSD"
CORPUS = BASE / "results" / "track4" / "corpus" / "CDSD"
OUT_DIR = BASE / "results" / "probe"

NON_PHONE = {"", "sil", "sp", "spn"}
TONE_LETTERS = set("˥˦˧˨˩")
_INTERVAL_RE = re.compile(
    r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*?)"'
)


def aligned_phones(tg_path: Path) -> list[tuple[float, float, str]]:
    """Return real (non-sil/spn) phone intervals from a TextGrid's phones tier."""
    try:
        text = tg_path.read_text(encoding="utf-8")
    except OSError:
        return []
    parts = text.split('name = "phones"')
    if len(parts) < 2:
        return []
    return [
        (float(a), float(b), lab)
        for a, b, lab in _INTERVAL_RE.findall(parts[1])
        if lab not in NON_PHONE
    ]


def pinyin_units(text: str):
    """Split a Mandarin transcript into Muthu-style initial / final / tone units.

    Returns (units_tone_sensitive, units_tone_agnostic, tone_sequence).
    """
    from pypinyin import Style, lazy_pinyin

    initials = lazy_pinyin(text, style=Style.INITIALS, strict=False)
    finals = lazy_pinyin(
        text, style=Style.FINALS_TONE3, strict=False, neutral_tone_with_five=True
    )

    toned: list[str] = []
    plain: list[str] = []
    tones: list[str] = []
    for ini, fin in zip(initials, finals):
        if not fin:
            continue
        tone = fin[-1] if fin[-1].isdigit() else ""
        base = fin[:-1] if tone else fin
        if not base:
            continue
        if ini:
            toned.append(ini)
            plain.append(ini)
        toned.append(base)
        plain.append(base)
        if tone:
            toned.append(f"T{tone}")
            tones.append(f"T{tone}")
    return toned, plain, tones


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=("1h", "10h", "all"), default="1h",
                    help="CDSD subset directory. Default 1h: it holds 55,851 utts across all 44 "
                         "speakers, matching the '~55,850 utterances' Muthu et al. report, whereas "
                         "10h holds 92,730 utts from only 8 speakers. Restricting also removes the "
                         "utt_id collisions, which occur only across the two subsets.")
    ap.add_argument("--per-speaker", type=int, default=200,
                    help="utterances sampled per speaker (0 = keep all)")
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--min-phones", type=int, default=1,
                    help="minimum real aligned phones; screens the ~14%% all-spn utterances")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "cdsd_probe_manifest.jsonl")
    args = ap.parse_args()

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    reasons = Counter()
    total = 0

    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if args.subset != "all" and Path(rec["audio_filepath"]).parent.name != args.subset:
                reasons[f"not_in_{args.subset}_subset"] += 1
                continue
            dur = float(rec.get("duration") or 0.0)
            if not (args.min_dur <= dur <= args.max_dur):
                reasons["duration_out_of_range"] += 1
                continue
            path = rec["audio_filepath"]
            if not os.path.isabs(path):
                path = str(BASE / path)
            if not os.path.exists(path):
                reasons["audio_missing"] += 1
                continue
            spk = rec["speaker_id"]
            utt_id = Path(rec["audio_filepath"]).stem
            tg = ALIGNED / spk / f"{utt_id}.TextGrid"
            phones = aligned_phones(tg)
            if not tg.exists():
                reasons["textgrid_missing"] += 1
                continue
            if len(phones) < args.min_phones:
                reasons["no_real_phones"] += 1
                continue
            # Integrity gate. `aligned/` and `corpus/` are keyed by (speaker, utt_id) only, but CDSD
            # reuses utt_ids across its 1h and 10h subsets: 6,103 (speaker, utt_id) pairs name two
            # different audio files with DIFFERENT transcripts. Only one .lab survives per key, so the
            # TextGrid at this path may belong to the other recording. Verified behaviour: for
            # non-colliding utterances the .lab matches the manifest transcript 400/400, so comparing
            # against it cleanly identifies the affected rows.
            lab = CORPUS / spk / f"{utt_id}.lab"
            try:
                lab_text = re.sub(r"\s+", "", lab.read_text(encoding="utf-8").strip())
            except OSError:
                reasons["lab_missing"] += 1
                continue
            if lab_text != re.sub(r"\s+", "", rec.get("text", "")):
                reasons["alignment_transcript_mismatch"] += 1
                continue
            toned, plain, tones = pinyin_units(rec.get("text", ""))
            if not toned:
                reasons["no_pinyin_units"] += 1
                continue
            by_speaker[spk].append(
                {
                    "uid": f"{spk}/{Path(rec['audio_filepath']).parent.name}/{utt_id}",
                    "utt_id": utt_id,
                    "subset": Path(rec["audio_filepath"]).parent.name,
                    "audio_filepath": path,
                    "speaker_id": spk,
                    "duration": dur,
                    "text": rec.get("text", ""),
                    "pseudo_severity": rec.get("pseudo_severity"),
                    "gender": rec.get("gender"),
                    "units_tone": toned,
                    "units_notone": plain,
                    "tones": tones,
                    "n_aligned_phones": len(phones),
                    "align_span_s": round(phones[-1][1] - phones[0][0], 3) if phones else 0.0,
                }
            )

    speakers = sorted(by_speaker)
    if not speakers:
        sys.exit("no usable utterances — check paths")

    rng = random.Random(args.seed)
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(0.70 * n)
    n_val = round(0.15 * n)
    split_of = {}
    for i, spk in enumerate(shuffled):
        split_of[spk] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    rows = []
    for spk in speakers:
        utts = sorted(by_speaker[spk], key=lambda r: r["utt_id"])
        if args.per_speaker and len(utts) > args.per_speaker:
            utts = random.Random(args.seed + hash(spk) % 10_000).sample(utts, args.per_speaker)
            utts.sort(key=lambda r: r["utt_id"])
        for r in utts:
            r["split"] = split_of[spk]
            rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report ----
    print(f"source utterances       : {total}")
    print("filtered out:")
    for k, v in reasons.most_common():
        print(f"    {k:24s} {v:7d}")
    print(f"speakers kept           : {len(speakers)}")
    print(f"utterances written      : {len(rows)}  -> {args.out}")

    per_split = Counter(r["split"] for r in rows)
    spk_split = Counter(split_of[s] for s in speakers)
    print("\nspeaker-disjoint split (70/15/15 over speakers):")
    for s in ("train", "val", "test"):
        print(f"    {s:6s} speakers={spk_split[s]:3d}  utts={per_split[s]:6d}")

    overlap = set()
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                sa = {s for s in speakers if split_of[s] == a}
                sb = {s for s in speakers if split_of[s] == b}
                overlap |= sa & sb
    print(f"    speaker overlap across splits: {len(overlap)} (must be 0)")

    sev = Counter(r["pseudo_severity"] for r in rows)
    print(f"\npseudo_severity: {dict(sev)}")
    tone_tokens = Counter(t for r in rows for t in r["tones"])
    tot = sum(tone_tokens.values()) or 1
    print("tone distribution (text-derived):")
    for t in sorted(tone_tokens):
        print(f"    {t}  {tone_tokens[t]:7d}  {100*tone_tokens[t]/tot:5.1f}%")
    units = Counter(u for r in rows for u in r["units_notone"])
    print(f"\ndistinct tone-agnostic units: {len(units)}   distinct tone-sensitive: "
          f"{len({u for r in rows for u in r['units_tone']})}")
    lens = [len(r["units_tone"]) for r in rows]
    print(f"units/utt: median={sorted(lens)[len(lens)//2]} max={max(lens)}")

    uids = Counter(r["uid"] for r in rows)
    dup = {k: v for k, v in uids.items() if v > 1}
    print(f"\nuid uniqueness: {len(uids)} unique / {len(rows)} rows  duplicates={len(dup)} (must be 0)")


if __name__ == "__main__":
    main()
