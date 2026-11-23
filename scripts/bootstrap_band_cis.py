#!/usr/bin/env python3
"""Speaker-level bootstrap CIs for the Phase-2 band sweep, on both corpora.

Why. The four-band comparison rests on 6 held-out speakers (CDSD) and 5 (UASpeech). Marginal CERs
alone cannot say whether `upper` vs `all` is a real ordering or sampling noise -- and the two corpora
disagree on that pair (CDSD favours `upper` by 1.47 pp, UASpeech favours `all` by 1.53 pp). This
script resamples SPEAKERS with replacement and recomputes CER per replicate, so the interval reflects
the unit that actually varies.

The decisive statistic is the PAIRED difference: within each replicate the same resampled speakers are
scored under both variants, so speaker-difficulty cancels. A paired CI that excludes zero is evidence
of a real ordering; one that straddles zero says the pair is unresolvable at this sample size.

Normalisation mirrors eval_lora_variants.py exactly, per corpus:
  CDSD     -- no case folding, character tokens (Mandarin convention).
  UASpeech -- case folded, character tokens for CER (word tokens are only used for WER upstream).

Usage:
    python scripts/layerprobe/bootstrap_band_cis.py --b 10000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
PROBE = BASE / "results" / "probe"

_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")

CORPORA = {
    "CDSD": {"dir": PROBE / "lora_preds", "case_fold": False},
    "UASpeech": {"dir": PROBE / "lora_preds_uaspeech", "case_fold": True},
}
VARIANTS = ["base", "upper", "middle", "all", "lower"]
# Pairs the paper makes claims about.
PAIRS = [("upper", "middle"), ("upper", "all"), ("all", "middle"), ("upper", "lower"),
         ("lower", "middle"),
         # Adaptation-vs-no-adaptation. §6.1 of the draft reports these as unresolved on CDSD and
         # names it the paper's weakest point; UASpeech has a far larger base->adapted gap.
         ("upper", "base"), ("all", "base"), ("middle", "base"), ("lower", "base")]


def norm(text: str, case_fold: bool) -> str:
    t = text or ""
    if case_fold:
        t = t.lower()
    return _PUNCT.sub("", t)


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


def load(corpus: str) -> tuple[list[str], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """-> (speakers, {variant: {speaker: (errs, refs)}}) with per-utterance counts."""
    cfg = CORPORA[corpus]
    per_variant: dict[str, dict[str, tuple[list, list]]] = {}
    speakers: set[str] = set()
    for v in VARIANTS:
        p = cfg["dir"] / f"{v}.test.preds.jsonl"
        if not p.exists():
            print(f"  [{corpus}] MISSING {p.name} -- skipping variant {v}")
            continue
        acc: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            spk = r["speaker_id"]
            ref = list(norm(r["ref"], cfg["case_fold"]))
            hyp = list(norm(r["hyp"], cfg["case_fold"]))
            acc[spk][0].append(_edit(ref, hyp))
            acc[spk][1].append(len(ref))
            speakers.add(spk)
        per_variant[v] = {s: (np.array(e), np.array(n)) for s, (e, n) in acc.items()}
    return sorted(speakers), per_variant


def cer_over(sel: list[str], data: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    e = sum(data[s][0].sum() for s in sel if s in data)
    n = sum(data[s][1].sum() for s in sel if s in data)
    return 100.0 * e / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=10000, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=PROBE / "band_bootstrap_cis.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_rows = []

    for corpus in CORPORA:
        speakers, per_variant = load(corpus)
        present = [v for v in VARIANTS if v in per_variant]
        print(f"\n=== {corpus}: {len(speakers)} test speakers, variants {present} ===")
        print(f"    speakers: {speakers}")

        # One shared set of speaker resamples, so every variant and every pair is scored on the
        # SAME replicates -- this is what makes the paired differences valid.
        idx = rng.integers(0, len(speakers), size=(args.b, len(speakers)))
        draws = [[speakers[i] for i in row] for row in idx]

        point = {v: cer_over(speakers, per_variant[v]) for v in present}
        boot = {v: np.array([cer_over(d, per_variant[v]) for d in draws]) for v in present}

        print(f"\n  {'variant':8s} {'CER':>7s}  {'95% CI':>18s}  {'width':>7s}")
        for v in present:
            lo, hi = np.percentile(boot[v], [2.5, 97.5])
            print(f"  {v:8s} {point[v]:7.2f}  [{lo:7.2f}, {hi:7.2f}]  {hi-lo:7.2f}")
            out_rows.append({"corpus": corpus, "kind": "marginal", "a": v, "b": "",
                             "point": round(point[v], 3), "lo": round(lo, 3), "hi": round(hi, 3),
                             "p_a_better": "", "n_speakers": len(speakers)})

        print(f"\n  paired differences (negative = first variant better)")
        print(f"  {'pair':22s} {'diff':>7s}  {'95% CI':>18s}  {'P(a<b)':>7s}  verdict")
        for a, b in PAIRS:
            if a not in present or b not in present:
                continue
            d = boot[a] - boot[b]
            dp = point[a] - point[b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_better = float((d < 0).mean())
            verdict = "RESOLVED" if (lo > 0 or hi < 0) else "unresolved (CI spans 0)"
            print(f"  {a+' - '+b:22s} {dp:7.2f}  [{lo:7.2f}, {hi:7.2f}]  {p_better:7.3f}  {verdict}")
            out_rows.append({"corpus": corpus, "kind": "paired", "a": a, "b": b,
                             "point": round(dp, 3), "lo": round(lo, 3), "hi": round(hi, 3),
                             "p_a_better": round(p_better, 4), "n_speakers": len(speakers)})

    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["corpus", "kind", "a", "b", "point", "lo", "hi",
                                           "p_a_better", "n_speakers"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
