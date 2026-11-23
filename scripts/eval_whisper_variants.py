#!/usr/bin/env python3
"""Score Whisper-small band adapters on the held-out CDSD test speakers.

Mirrors eval_lora_variants.py (the Qwen arm) so the two sets of numbers are produced under the same
scoring conventions: same manifest, same split, same normalisation, same CER definition.

Mandarin convention, matching the Qwen arm's defaults: no case folding, character tokens, and CER is
the only metric quoted -- a character-tokenised edit rate IS the character error rate, so it is not
labelled "WER" here (see §4 of the paper).

No merge step: the PEFT adapter is loaded onto the frozen base and used directly for generation. That
avoids writing four ~1 GB merged checkpoints for numbers that do not depend on them.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/eval_whisper_variants.py \
        --variant base=NONE \
        --variant middle=models/layerprobe_lora_whisper/middle__<sig>/adapter \
        --out results/probe/whisper_band_sweep.csv
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
import soundfile as sf
import torch

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
SR = 16000
_PUNCT = re.compile(r"[\s，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")
_PUNCT_KEEPSPACE = re.compile(r"[，。！？、；：“”‘’（）《》,.!?;:\"'()\[\]<>—-]+")

# Set from --case-fold / --tokenise. English needs both: Whisper emits capitalised, punctuated text
# while the UASpeech references are lowercase bare words, so without case folding the score measures
# capitalisation. These MUST match the settings used for the Qwen3-ASR arm in eval_lora_variants.py,
# or the cross-model comparison is scored on two different conventions.
CASE_FOLD = False
TOKENISE = "char"


def norm(text: str) -> str:
    t = text or ""
    if CASE_FOLD:
        t = t.lower()
    return _PUNCT.sub("", t)


def norm_words(text: str) -> list[str]:
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
    r, h = norm_words(ref), norm_words(hyp)
    return _edit(r, h), len(r)


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torchaudio
        audio = (torchaudio.transforms.Resample(sr, SR)(torch.from_numpy(audio).unsqueeze(0))
                 .squeeze(0).numpy())
    return audio


# --- decode configuration -------------------------------------------------
# The original setting (greedy, flat max_new_tokens=200) let Whisper-small run repetition loops on
# short utterances: every looped character scores as an insertion, which is why unadapted CER came out
# above 100%. A fixed small cap is not the fix either - CDSD reaches 18.3 s and UASpeech 19.5 s, so a
# flat 48 would truncate the tail and trade an insertion artefact for a deletion one. The cap is
# therefore scaled by audio duration. Batched generation shares one bound, so the batch MAXIMUM sets
# it; the cap is only an upper bound, and shorter utterances still stop at EOS on their own.
DECODE = "fixed"
_TOK_PER_SEC = 16      # generous for Mandarin chars (~4-5/s) and English words alike
_SLACK = 12
_CAP_MIN, _CAP_MAX = 24, 200


def gen_kw(chunk: list) -> dict:
    """Generation kwargs for one batch. `published` reproduces the original artefact-carrying decode."""
    if DECODE == "published":
        return dict(max_new_tokens=_CAP_MAX)
    durs = [r.get("duration") or 0.0 for r in chunk]
    longest = max(durs) if durs else 0.0
    cap = _CAP_MAX if longest <= 0 else int(longest * _TOK_PER_SEC + _SLACK)
    cap = max(_CAP_MIN, min(_CAP_MAX, cap))
    return dict(max_new_tokens=cap, no_repeat_ngram_size=3, num_beams=5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True,
                    help="name=adapter_dir, or name=NONE for the unadapted base")
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_lora_manifest.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--language", default="chinese")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--decode", choices=("fixed", "published"), default="fixed",
                    help="'fixed' = duration-scaled cap + no_repeat_ngram_size 3 + beam 5; "
                         "'published' = the original greedy max_new_tokens=200 (kept reproducible)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "whisper_band_sweep.csv")
    ap.add_argument("--preds-dir", type=Path, default=PROBE_DIR / "whisper_preds")
    ap.add_argument("--case-fold", action="store_true",
                    help="Lowercase both sides before scoring. REQUIRED for alphabetic languages; "
                         "must match the setting used for the Qwen3-ASR arm.")
    ap.add_argument("--tokenise", choices=("char", "word"), default="char",
                    help="Unit for the reported error rate. UASpeech is isolated words, so word is "
                         "the natural primary metric; CER is reported alongside either way.")
    args = ap.parse_args()
    global DECODE
    DECODE = args.decode
    print(f"decode mode: {DECODE}")

    # The flags must reach the module globals BEFORE any scoring happens; a --case-fold that is parsed
    # but never applied would silently score capitalisation and look like a real model difference.
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
    if not rows:
        raise SystemExit(f"no rows with split={args.split} in {args.manifest}")

    variants = []
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant expects name=path, got {spec!r}")
        name, path = spec.split("=", 1)
        if path != "NONE" and not Path(path).exists():
            raise SystemExit(f"adapter path does not exist: {path}")
        variants.append((name, path))

    args.preds_dir.mkdir(parents=True, exist_ok=True)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(args.model)
    forced = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")

    results = []
    for name, path in variants:
        print(f"\n=== {name} ({path}) ===")
        model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
        if path != "NONE":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, path)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

        hyps: dict[str, str] = {}
        t0 = time.time()
        pred_path = args.preds_dir / f"{name}.{args.split}.preds.jsonl"
        with open(pred_path, "w", encoding="utf-8") as fh:
            for i in range(0, len(rows), args.batch_size):
                chunk = rows[i:i + args.batch_size]
                try:
                    audio = [load_audio(r["audio_filepath"]) for r in chunk]
                    feats = processor(audio, sampling_rate=SR, return_tensors="pt").input_features
                    feats = feats.to(model.device, dtype=torch.bfloat16)
                    with torch.no_grad():
                        ids = model.generate(feats, forced_decoder_ids=forced, **gen_kw(chunk))
                    texts = processor.batch_decode(ids, skip_special_tokens=True)
                except Exception as exc:  # noqa: BLE001 -- one bad batch must not kill the pass
                    print(f"  !! batch {i}: {type(exc).__name__}: {exc} — retrying singly")
                    texts = []
                    for r in chunk:
                        try:
                            a = load_audio(r["audio_filepath"])
                            f = processor(a, sampling_rate=SR, return_tensors="pt").input_features
                            f = f.to(model.device, dtype=torch.bfloat16)
                            with torch.no_grad():
                                one = model.generate(f, forced_decoder_ids=forced, **gen_kw([r]))
                            texts.append(processor.batch_decode(one, skip_special_tokens=True)[0])
                        except Exception as exc2:  # noqa: BLE001
                            print(f"     !! {r['uid']}: {type(exc2).__name__}: {exc2}")
                            texts.append("")
                for r, t in zip(chunk, texts):
                    hyps[r["uid"]] = t
                    fh.write(json.dumps({"uid": r["uid"], "speaker_id": r["speaker_id"],
                                         "severity": r.get("pseudo_severity"),
                                         "ref": r["text"], "hyp": t}, ensure_ascii=False) + "\n")
                if i % (args.batch_size * 20) == 0:
                    done = min(i + args.batch_size, len(rows))
                    print(f"  {done}/{len(rows)}  {done/max(time.time()-t0,1e-6):.1f} utt/s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ce = ct = we = wt = 0
        by_sev: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_spk: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            e, n = cer_counts(r["text"], hyps.get(r["uid"], ""))
            ce += e
            ct += n
            ew, nw = wer_counts(r["text"], hyps.get(r["uid"], ""))
            we += ew
            wt += nw
            sev = r.get("pseudo_severity") or "unknown"
            by_sev[sev][0] += e
            by_sev[sev][1] += n
            by_spk[r["speaker_id"]][0] += e
            by_spk[r["speaker_id"]][1] += n
        cer = 100.0 * ce / max(ct, 1)
        wer = 100.0 * we / max(wt, 1)
        sev_cer = {k: round(100.0 * v[0] / max(v[1], 1), 2) for k, v in sorted(by_sev.items())}
        spk_cer = {k: round(100.0 * v[0] / max(v[1], 1), 2) for k, v in sorted(by_spk.items())}
        print(f"  CER {cer:.3f}  WER {wer:.3f}   per-difficulty {sev_cer}")
        results.append({"variant": name, "path": path, "n": len(rows), "cer": round(cer, 3),
                        "wer": round(wer, 3),
                        "per_difficulty_cer": sev_cer, "per_speaker_cer": spk_cer,
                        "elapsed_s": round(time.time() - t0, 1)})

    print(f"\n{'variant':10s} {'CER%':>8s}")
    for r in results:
        print(f"{r['variant']:10s} {r['cer']:8.3f} {r['wer']:8.3f}")

    import csv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "path", "n", "cer", "wer",
                                           "per_difficulty_cer", "per_speaker_cer", "elapsed_s"])
        w.writeheader()
        for r in results:
            w.writerow({**r,
                        "per_difficulty_cer": json.dumps(r["per_difficulty_cer"], ensure_ascii=False),
                        "per_speaker_cer": json.dumps(r["per_speaker_cer"], ensure_ascii=False)})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
