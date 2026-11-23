#!/usr/bin/env python3
"""Score Phase-2 LoRA variants on the held-out CDSD test speakers.

Decodes with Qwen's own `qwen_asr.Qwen3ASRModel.transcribe` — the path every published Track 1 number
uses, and the only one validated here (see merge_adapter_export.py docstring for the 41.87 / 124.57 /
32.38 comparison that ruled out the generic transformers generate path).

Reports, per variant: CER, WER, and CER relative to the all-layer reference, plus a per-severity
breakdown. The headline comparison is `upper` (17-24) vs `middle` (9-16) — Probe B predicted the upper
band wins on Qwen3-ASR, the reverse of Muthu et al.'s Whisper-small result.

Runs in the MAIN venv (qwen_asr lives there; it bundles its own modelling code and works under
transformers 4.57.6).

Usage (DGX, MAIN venv):
    source venv/bin/activate
    python scripts/layerprobe/eval_lora_variants.py \
        --variant base=Qwen/Qwen3-ASR-1.7B \
        --variant upper=models/layerprobe_lora/upper__734ee1d612/merged \
        --variant middle=models/layerprobe_lora/middle__<sig>/merged \
        --variant all=models/layerprobe_lora/all__<sig>/merged
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")
# Same class minus \s, for word-tokenised scoring where whitespace is the token delimiter.
_PUNCT_KEEPSPACE = re.compile(r"[，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")

# Set by main() from --case-fold / --tokenise. Defaults reproduce the original Mandarin behaviour.
CASE_FOLD = False
TOKENISE = "char"


def norm(text: str) -> str:
    t = text or ""
    if CASE_FOLD:
        t = t.lower()
    return _PUNCT.sub("", t)


def norm_words(text: str) -> list[str]:
    """Whitespace-delimited tokens with punctuation stripped — real WER for alphabetic languages."""
    t = text or ""
    if CASE_FOLD:
        t = t.lower()
    return _PUNCT_KEEPSPACE.sub(" ", t).split()


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


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(norm(ref)), list(norm(hyp))
    return _edit(r, h), len(r)


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    if TOKENISE == "word":
        r, h = norm_words(ref), norm_words(hyp)
    else:
        # Mandarin has no orthographic word boundaries; treat each character as a token so the number
        # is defined, and report CER as the primary metric (cf. track1/CLAUDE.md: zh WER is a
        # tokeniser artefact).
        r, h = list(norm(ref)), list(norm(hyp))
    return _edit(r, h), len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True,
                    help="name=path/to/merged (repeatable); 'base' may point at the HF id")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_lora_manifest.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "qwen3asr_zh_target_sweep.csv")
    ap.add_argument("--preds-dir", type=Path, default=PROBE_DIR / "lora_preds")
    ap.add_argument("--case-fold", action="store_true",
                    help="Lowercase both sides before scoring. REQUIRED for alphabetic languages: "
                         "Qwen3-ASR emits capitalised text while UASpeech refs are lowercase, so "
                         "without this every initial letter counts as a substitution.")
    ap.add_argument("--tokenise", choices=("char", "word"), default="char",
                    help="WER token unit. 'char' (default) reproduces the Mandarin convention where "
                         "WER==CER; 'word' gives real word-level WER for alphabetic languages.")
    args = ap.parse_args()

    global CASE_FOLD, TOKENISE
    CASE_FOLD = args.case_fold
    TOKENISE = args.tokenise
    print(f"scoring: case_fold={CASE_FOLD} tokenise={TOKENISE} language={args.language}")

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    speakers = sorted({r["speaker_id"] for r in rows})
    print(f"eval split={args.split}: {len(rows)} utts / {len(speakers)} speakers {speakers}")

    variants = []
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        variants.append((name, path))

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    from qwen_asr import Qwen3ASRModel

    results = []
    for name, path in variants:
        pred_path = args.preds_dir / f"{name}.{args.split}.preds.jsonl"
        print(f"\n=== {name} ({path}) ===")
        asr = Qwen3ASRModel.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")

        hyps: dict[str, str] = {}
        t0 = time.time()
        with open(pred_path, "w", encoding="utf-8") as fh:
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i:i + args.batch_size]
                paths = [r["audio_filepath"] for r in chunk]
                try:
                    outs = asr.transcribe(audio=paths, language=args.language)
                    texts = [(o.text if o is not None else "") for o in outs]
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! batch {i}: {type(exc).__name__}: {exc} — retrying singly")
                    texts = []
                    for r in chunk:
                        try:
                            o = asr.transcribe(audio=r["audio_filepath"], language=args.language)
                            texts.append(o[0].text if o else "")
                        except Exception as exc2:  # noqa: BLE001
                            print(f"     !! {r['uid']}: {type(exc2).__name__}: {exc2}")
                            texts.append("")
                for r, t in zip(chunk, texts):
                    if "<asr_text>" in t:
                        t = t.split("<asr_text>", 1)[1]
                    hyps[r["uid"]] = t
                    fh.write(json.dumps({"uid": r["uid"], "speaker_id": r["speaker_id"],
                                         "severity": r.get("pseudo_severity"),
                                         "ref": r["text"], "hyp": t}, ensure_ascii=False) + "\n")
                if i % (args.batch_size * 40) == 0:
                    done = min(i + args.batch_size, len(rows))
                    print(f"  {done}/{len(rows)}  {done/max(time.time()-t0,1e-6):.1f} utt/s")

        del asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ce = ct = we = wt = 0
        by_sev: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_spk: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            h = hyps.get(r["uid"], "")
            e, n = cer_counts(r["text"], h)
            ce += e
            ct += n
            e2, n2 = wer_counts(r["text"], h)
            we += e2
            wt += n2
            sev = r.get("pseudo_severity") or "unknown"
            by_sev[sev][0] += e
            by_sev[sev][1] += n
            by_spk[r["speaker_id"]][0] += e
            by_spk[r["speaker_id"]][1] += n

        cer = 100.0 * ce / max(ct, 1)
        wer = 100.0 * we / max(wt, 1)
        sev_cer = {k: round(100.0 * v[0] / max(v[1], 1), 2) for k, v in sorted(by_sev.items())}
        spk_cer = {k: round(100.0 * v[0] / max(v[1], 1), 2) for k, v in sorted(by_spk.items())}
        results.append({"variant": name, "path": path, "n": len(rows), "cer": round(cer, 3),
                        "wer": round(wer, 3), "per_severity_cer": sev_cer,
                        "per_speaker_cer": spk_cer, "elapsed_s": round(time.time() - t0, 1)})
        print(f"  CER {cer:.3f}  WER {wer:.3f}   per-severity {sev_cer}")

    ref = next((r for r in results if r["variant"] == "all"), None)
    print(f"\n{'variant':10s} {'CER%':>8s} {'WER%':>8s} {'rel. vs all':>12s}")
    for r in results:
        rel = ""
        if ref and ref["cer"]:
            rel = f"{100*(r['cer']-ref['cer'])/ref['cer']:+.2f}%"
        r["cer_rel_vs_all_pct"] = rel
        print(f"{r['variant']:10s} {r['cer']:8.3f} {r['wer']:8.3f} {rel:>12s}")

    import csv

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "path", "n", "cer", "wer",
                                           "cer_rel_vs_all_pct", "per_severity_cer",
                                           "per_speaker_cer", "elapsed_s"])
        w.writeheader()
        for r in results:
            w.writerow({**r, "per_severity_cer": json.dumps(r["per_severity_cer"],
                                                            ensure_ascii=False),
                        "per_speaker_cer": json.dumps(r["per_speaker_cer"], ensure_ascii=False)})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
