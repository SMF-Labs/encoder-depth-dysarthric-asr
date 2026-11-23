#!/usr/bin/env python3
"""Whisper-small band sweep on OUR CDSD split, gate and scoring.

Why this exists. The paper's central comparison is currently cross-PAPER: our Qwen3-ASR band ordering
against a published Whisper-small result obtained with a different implementation, different
preprocessing, a different effective dataset, and no equivalent alignment-integrity gate (we drop 14.6%
of utterances they did not). That makes "the depth profile differs between these encoders" an inference
across two experimental setups rather than a controlled measurement. Running Whisper-small through the
IDENTICAL pipeline removes every one of those confounds except the architecture itself.

The prediction under test, registered before any Whisper adapter is trained: on Whisper-small the
MIDDLE band (blocks 5-8) should beat the UPPER band (9-12), reversing the Qwen3-ASR ordering. That is
what the prior work reports and what our knee-rule hypothesis predicts, since Whisper-small's phoneme
error curve is reported to plateau at ~50% of relative depth.

Band definition is proportional, not absolute. Qwen3-ASR has 24 encoder blocks and its bands are 8
blocks each; Whisper-small has 12, so its bands are 4 blocks each:
    lower  = 1-4    middle = 5-8 (the prior work's winning band)    upper = 9-12    all = 1-12

Every data gate is IMPORTED from train_lora_target_sweep rather than reimplemented, so the manifest
schema check, unknown-split rejection, non-empty split check, speaker-leak rejection, corpus
compose-check, bounded per-utterance drop budget and audio preflight are literally the same code that
audited PASS for the Qwen runs. Only the model, the LoRA target pattern and the collate differ.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/train_lora_whisper_sweep.py --variant middle --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora_target_sweep import (  # noqa: E402  -- audited gates, imported not copied
    CDSDDataset,
    CORPUS_PRESETS,
    MAX_DROP_FRAC_CEILING,
    load_and_validate,
)

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
N_ENCODER_BLOCKS = 12          # whisper-small
BLOCKS_PER_BAND = 4            # 12 / 3, the proportional analogue of Qwen's 24 / 3

# 1-based inclusive block ranges, converted to 0-based indices below.
# Whisper's decoder prefix is language-conditioned, so the language is a property of the corpus and
# must not be settable independently: --corpus uaspeech with --language chinese would train English
# audio under a Chinese prefix and silently invalidate that arm.
CORPUS_LANGUAGE = {"cdsd": "chinese", "uaspeech": "english"}

BANDS = {
    "lower": list(range(0, 4)),      # blocks 1-4
    "middle": list(range(4, 8)),     # blocks 5-8  <- prior work's winning band
    "upper": list(range(8, 12)),     # blocks 9-12
    "all": list(range(0, 12)),
}


def lora_target_regex(layer_idx: list[int]) -> str:
    """Encoder self-attention projections only, for the given 0-based blocks."""
    sel = "|".join(str(i) for i in sorted(layer_idx))
    return rf"^model\.encoder\.layers\.({sel})\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"


def build_collate(processor, tokenizer, decoder_start_token_id: int):
    """Whisper collate: log-mel input_features + label ids padded with -100.

    The tokenizer emits <|startoftranscript|> as token 0, and WhisperForConditionalGeneration
    ALSO prepends decoder_start_token_id when it right-shifts the labels to build decoder inputs.
    Leaving both duplicates the start token and shifts the optimisation target by one position --
    silently, with no error and no crash, just worse training. Strip the leading token when it is
    the decoder start token, which is the standard Whisper fine-tuning recipe.
    """
    def collate(batch):
        audio = [b["audio"] for b in batch]
        feats = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features
        enc = tokenizer([b["text"] for b in batch], return_tensors="pt",
                        padding=True, truncation=True, max_length=448)
        labels, attn = enc.input_ids, enc.attention_mask
        if labels.shape[1] > 1 and bool((labels[:, 0] == decoder_start_token_id).all()):
            labels, attn = labels[:, 1:], attn[:, 1:]
        # Mask padding via the ATTENTION MASK, not by comparing to pad_token_id. Whisper's pad token
        # is the SAME id as its end-of-transcript token, so value-equality masking would also delete
        # every genuine terminal EOT from the loss and the model would never learn to stop -- while
        # the Qwen arm trains its EOS explicitly. That asymmetry would be an accidental confound in
        # a comparison whose entire purpose is to isolate the architecture.
        labels = labels.masked_fill(attn == 0, -100)
        return {"input_features": feats, "labels": labels}
    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=tuple(BANDS))
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--manifest", type=Path,
                    default=BASE / "results" / "probe" / "cdsd_lora_manifest.jsonl")
    ap.add_argument("--language", default=None,
                    help="Derived from --corpus; pass only to assert the expected value. A value "
                         "disagreeing with the corpus is rejected.")
    ap.add_argument("--corpus", choices=tuple(CORPUS_PRESETS), default="cdsd",
                    help="expected-composition preset, as in the Qwen arm")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="Matches the audited Qwen recipe. That value is an OOM safety cap for "
                         "long dysarthric clips, and keeping it identical preserves gradient-"
                         "accumulation order across the two architectures being compared.")
    ap.add_argument("--effective-batch", type=int, default=128)
    ap.add_argument("--min-train", type=int, default=-1)
    ap.add_argument("--preflight", type=int, default=2000)
    ap.add_argument("--compose-tolerance", type=float, default=0.02)
    ap.add_argument("--max-drop-frac", type=float, default=0.02)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=BASE / "models" / "layerprobe_lora_whisper")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Same fail-fast argument validation as the Qwen trainer; see its rationale comment.
    if args.epochs <= 0:
        raise SystemExit(f"--epochs must be > 0, got {args.epochs}")
    if not math.isfinite(args.compose_tolerance) or not (0 <= args.compose_tolerance < 1):
        raise SystemExit(f"--compose-tolerance must be finite and in [0, 1), got {args.compose_tolerance}")
    if args.max_train < 0:
        raise SystemExit(f"--max-train must be >= 0 (0 = no cap), got {args.max_train}")
    if not (0 <= args.seed < 2**32):
        raise SystemExit(f"--seed must be in [0, 2**32), got {args.seed}")
    if not (0 < args.max_drop_frac <= MAX_DROP_FRAC_CEILING):
        raise SystemExit(f"--max-drop-frac must be in (0, {MAX_DROP_FRAC_CEILING}], "
                         f"got {args.max_drop_frac}")
    if args.effective_batch % args.batch_size:
        raise SystemExit(f"--effective-batch {args.effective_batch} not divisible by "
                         f"--batch-size {args.batch_size}")
    for name, v in (("--rank", args.rank), ("--alpha", args.alpha),
                    ("--batch-size", args.batch_size), ("--effective-batch", args.effective_batch),
                    ("--preflight", args.preflight)):
        if v < 0 or (name != "--preflight" and v <= 0):
            raise SystemExit(f"{name} must be positive, got {v}")
    if not (0 <= args.dropout < 1):
        raise SystemExit(f"--dropout must be in [0, 1), got {args.dropout}")
    if not (math.isfinite(args.lr) and args.lr > 0):
        raise SystemExit(f"--lr must be finite and > 0, got {args.lr}")
    if not torch.cuda.is_available() and not os.environ.get("LAYERPROBE_ALLOW_CPU"):
        raise SystemExit("no CUDA device visible — refusing to start a multi-hour run on CPU "
                         "(set LAYERPROBE_ALLOW_CPU=1 to override)")
    grad_acc = args.effective_batch // args.batch_size

    idx = BANDS[args.variant]
    print("--- pre-launch check ---")
    print(f"  arch            : whisper-small, {N_ENCODER_BLOCKS} encoder blocks")
    print(f"  variant         : {args.variant}  blocks {idx[0]+1}-{idx[-1]+1} (0-based {idx})")
    print(f"  success criteria: adapter weights written; completion gate >= "
          f"{100*(1-args.max_drop_frac):.0f}% of the training split readable; exact "
          f"{4*len(idx)} encoder self-attn modules targeted and nothing trainable outside the encoder")
    print(f"  running signals : (1) loss every 25 steps, (2) per-epoch checkpoint, "
          f"(3) skipped-utterance counter")
    print(f"  prediction under test: on whisper-small, MIDDLE (5-8) should beat UPPER (9-12), "
          f"reversing the Qwen3-ASR ordering")
    print("--- end pre-launch check ---")

    # ---- audited data gates, imported from the Qwen trainer ----
    # Resolve the -1 sentinel to 90% of the documented CDSD train size, exactly as the Qwen arm
    # does. Passing -1 straight through would make the imported `len(train) < min_train` check
    # unable to fire, i.e. an imported gate that is present but vacuous.
    _exp_spk, _exp_rows, _exp_train, _corpus_dir, _subset = CORPUS_PRESETS[args.corpus]
    _lang = CORPUS_LANGUAGE.get(args.corpus)
    if _lang is None:
        raise SystemExit(f"no language mapping for --corpus {args.corpus}; add one to CORPUS_LANGUAGE "
                         f"before running this corpus through the Whisper arm")
    if args.language is not None and args.language != _lang:
        raise SystemExit(f"--language {args.language!r} disagrees with --corpus {args.corpus} "
                         f"(expects {_lang!r}); the decoder prefix is language-conditioned, so this "
                         f"combination would train under the wrong prefix")
    args.language = _lang
    print(f"  language        : {args.language} (derived from --corpus {args.corpus})")
    if args.min_train == -1:
        min_train = int(0.9 * _exp_train)
    elif args.min_train <= 0:
        raise SystemExit(f"--min-train must be > 0 (or -1 for the 90% default), got {args.min_train}")
    else:
        min_train = args.min_train
    print(f"  min-train gate  : {min_train} rows")
    # NB return order is (train, val, test, audit) — not audit-first.
    train, val, test, audit = load_and_validate(
        args.manifest, min_train, args.preflight, args.compose_tolerance,
        exp_speakers=_exp_spk, exp_rows=_exp_rows, exp_train=_exp_train,
        corpus_dir=_corpus_dir, subset=_subset)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"manifest sha={audit['manifest_sha256'][:16]}…")
    if args.max_train:
        train = train[: args.max_train]

    # Hash BOTH files: this script's behaviour depends on the gates and dataset imported from
    # train_lora_target_sweep, so an edit there must produce a different run directory rather than
    # silently reusing one. Gate-affecting arguments enter the signature for the same reason.
    code_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + (Path(__file__).parent / "train_lora_target_sweep.py").read_bytes()
    ).hexdigest()[:8]
    sig_payload = {"model": args.model, "layers": idx, "rank": args.rank, "alpha": args.alpha,
                   "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
                   "epochs": args.epochs, "eff_batch": args.effective_batch,
                   "batch_size": args.batch_size, "max_train": args.max_train,
                   "manifest_sha256": audit["manifest_sha256"], "code_sha256": code_hash,
                   "seed": args.seed, "language": args.language, "min_train": min_train,
                   "preflight": args.preflight, "compose_tolerance": args.compose_tolerance,
                   "max_drop_frac": args.max_drop_frac, "corpus": args.corpus}
    sig = hashlib.sha256(json.dumps(sig_payload, sort_keys=True).encode()).hexdigest()[:10]
    out_dir = args.out_root / f"{args.variant}__{sig}"
    adapter_dir = out_dir / "adapter"
    if adapter_dir.exists() and not (args.overwrite or args.resume):
        raise SystemExit(f"{adapter_dir} exists — pass --resume or --overwrite")
    if args.overwrite and args.resume:
        raise SystemExit("--overwrite and --resume are contradictory; pass exactly one")
    if args.overwrite and out_dir.exists():
        # Clear stale checkpoints and metadata; leaving them lets a later --resume pick up a
        # checkpoint that belongs to a different run configuration.
        import shutil as _sh
        _sh.rmtree(out_dir)
        print(f"  --overwrite: cleared {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import shutil as _shutil
    _MIN_FREE_GB = float(os.environ.get("LAYERPROBE_MIN_FREE_GB", "25"))
    if not math.isfinite(_MIN_FREE_GB) or _MIN_FREE_GB < 0:
        raise SystemExit(f"LAYERPROBE_MIN_FREE_GB must be finite and >= 0, got {_MIN_FREE_GB}")
    free_gb = _shutil.disk_usage(out_dir).free / 1e9
    if free_gb < _MIN_FREE_GB:
        raise SystemExit(f"only {free_gb:.1f} GB free at {out_dir} — need >= {_MIN_FREE_GB:.0f} GB")
    print(f"  disk free       : {free_gb:.1f} GB")

    (out_dir / "_speaker_disjoint_audit.json").write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (out_dir / "_run_config.json").write_text(
        json.dumps({**sig_payload, "variant": args.variant, "run_sig": sig,
                    "arch": "whisper-small", "n_encoder_blocks": N_ENCODER_BLOCKS}, indent=2),
        encoding="utf-8")

    # ---- seed BEFORE model construction and PEFT init (see Qwen trainer's comment) ----
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor, WhisperTokenizer, set_seed)
    set_seed(args.seed)
    print(f"  seeded (model init + data order): {args.seed}")

    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16)
    processor = WhisperProcessor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model, language=args.language,
                                                 task="transcribe")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    n_blocks = len(model.model.encoder.layers)
    if n_blocks != N_ENCODER_BLOCKS:
        raise SystemExit(f"expected {N_ENCODER_BLOCKS} encoder blocks in {args.model}, found {n_blocks} "
                         f"— band definitions assume whisper-small")

    from peft import LoraConfig, get_peft_model
    rx = lora_target_regex(idx)
    matched = [n for n, _ in model.named_modules() if re.match(rx, n)]
    expected = 4 * len(idx)
    if len(matched) != expected:
        raise SystemExit(f"LoRA target regex matched {len(matched)} modules, expected {expected}: "
                         f"{matched[:6]}")
    if any(".encoder." not in n for n in matched):
        raise SystemExit(f"non-encoder module matched: {[n for n in matched if '.encoder.' not in n][:5]}")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha,
                                             lora_dropout=args.dropout, bias="none",
                                             target_modules=rx, task_type=None))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("no trainable parameters after get_peft_model")
    outside = [n for n, _ in trainable if ".encoder." not in n]
    if outside:
        raise SystemExit(f"trainable parameters outside the encoder: {outside[:5]}")
    for _, p in trainable:
        p.data = p.data.float()          # fp32 master weights on the low-rank params
    n_train_p = sum(p.numel() for _, p in trainable)
    print(f"trainable LoRA params: {n_train_p:,} over {len(matched)} modules")

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=grad_acc,
        learning_rate=args.lr, weight_decay=args.weight_decay, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, logging_steps=25, save_strategy="epoch",
        save_total_limit=2, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, seed=args.seed, data_seed=args.seed,
    )
    torch.backends.cuda.enable_cudnn_sdp(False)

    ds = CDSDDataset(train, max_failures=int(args.max_drop_frac * len(train)))
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds,
                             data_collator=build_collate(
                                 processor, tokenizer, model.config.decoder_start_token_id))

    # ---- enforced runtime smoke gate (ML-04), under the same autocast the Trainer uses ----
    if not args.resume:
        b = next(iter(trainer.get_train_dataloader()))
        b = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in b.items()}
        model.train()
        with torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                            dtype=torch.bfloat16):
            out = model(**b)
        if out.loss is None or not torch.isfinite(out.loss):
            raise SystemExit(f"smoke step produced non-finite loss ({out.loss})")
        out.loss.backward()
        n_grad = sum(1 for _, p in trainable if p.grad is not None)
        if n_grad == 0:
            raise SystemExit("smoke step produced no gradients on any trainable parameter")
        model.zero_grad(set_to_none=True)
        print(f"  smoke step OK   : loss={out.loss.item():.4f}, {n_grad} trainable tensors have grads")
        del out, b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    resume = None
    if args.resume:
        # Inventory checkpoints and select the deepest, failing loudly if none exists, rather than
        # letting a bare boolean silently restart from scratch.
        ckpts = [(int(m.group(1)), d) for d in out_dir.iterdir() if d.is_dir()
                 for m in [re.match(r"^checkpoint-(\d+)$", d.name)] if m]
        if not ckpts:
            raise SystemExit(f"--resume passed but no checkpoint-N directory under {out_dir}")
        step, resume = max(ckpts, key=lambda t: t[0])
        resume = str(resume)
        print(f"  resuming from   : {resume} (step {step})")

    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(adapter_dir))

    written = list(adapter_dir.glob("*"))
    if not any(f.suffix in (".safetensors", ".bin") for f in written):
        raise SystemExit(f"no adapter weights written to {adapter_dir}: {[f.name for f in written]}")
    skipped = len(ds.failures)
    tol = int(args.max_drop_frac * len(train))
    print(f"completion: {100*(len(train)-skipped)/max(len(train),1):.2f}% of the training split "
          f"readable ({skipped} skipped, tolerance {tol})")
    if skipped > tol:
        raise SystemExit(f"completion gate failed: {skipped} unreadable > tolerance {tol}")
    (out_dir / "_run_result.json").write_text(
        json.dumps({"variant": args.variant, "seed": args.seed, "skipped": skipped,
                    "n_train": len(train), "trainable_params": n_train_p}, indent=2),
        encoding="utf-8")
    print(f"{args.variant}: adapter -> {adapter_dir}")


if __name__ == "__main__":
    main()
