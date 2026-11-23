#!/usr/bin/env python3
"""Phase 2 — LoRA target-layer sweep on the Qwen3-ASR audio encoder.

Tests whether Muthu et al.'s "adapt the middle encoder layers" result transfers to Qwen3-ASR. Their
justification is that phoneme information matures mid-encoder (their layers 5-8 of 12). CANDOR's Probe B
sweep found the opposite on this architecture: PER keeps improving to ~layer 20-21 of 24, i.e. ~85%
depth, gaining a further 15.9 pp beyond the midpoint on the (uncontaminated) base model. So the probe
predicts the best band here is UPPER (17-24), not the proportional analogue of theirs (9-16).

Variants (encoder-only LoRA, decoder frozen, matching the paper's setup):
    all        all 24 encoder layers               <- their "All"
    single_L   one layer, L in {2,6,10,14,18,22}   <- 14 is the analogue of their best single layer 7
    lower      layers 1-8                          <- analogue of their 1-4
    middle     layers 9-16                         <- analogue of their winning 5-8
    upper      layers 17-24                        <- analogue of their 9-12; the probe predicts this wins

LoRA config follows the paper: q_proj + k_proj + v_proj + out_proj, r=64, alpha=128, dropout 0.1;
AdamW lr 3e-5, effective batch 128, 6 epochs, cosine schedule, weight decay 0.03.

Data is the CDSD 1h subset (the subset Muthu used: their "~55,850 utterances" == 1h's 55,851), on the
same speaker-disjoint 70/15/15 split as the probe so probe and adaptation results are commensurable.

Adapts BASE Qwen/Qwen3-ASR-1.7B, not the zh-aug fine-tune: zh-aug already trained on 81.9% of these
utterances and 37/44 speakers including all 6 test speakers, so adapting it would measure nothing.

MUST load through qwen3asr_compat — a plain from_pretrained silently gives a random model on these
checkpoints. See that module's docstring.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/train_lora_target_sweep.py --variant middle
    python scripts/layerprobe/train_lora_target_sweep.py --variant upper --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3asr_compat import load_qwen3_asr  # noqa: E402

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
N_ENC_LAYERS = 24
SR = 16000
EXPECTED_SPEAKERS = 44  # CDSD 1h subset; compose-check guards against a truncated manifest


def layers_for(variant: str) -> list[int]:
    if variant == "all":
        return list(range(N_ENC_LAYERS))
    if variant == "lower":
        return list(range(0, 8))
    if variant == "middle":
        return list(range(8, 16))
    if variant == "upper":
        return list(range(16, 24))
    m = re.fullmatch(r"single_(\d+)", variant)
    if m:
        idx = int(m.group(1))
        if not 1 <= idx <= N_ENC_LAYERS:
            raise SystemExit(f"layer {idx} out of range 1..{N_ENC_LAYERS}")
        return [idx - 1]  # 1-based in the paper's tables, 0-based in the module list
    raise SystemExit(f"unknown variant: {variant}")


def lora_target_regex(layer_idx: list[int]) -> str:
    """Regex over module names selecting q/k/v/out_proj within the chosen encoder layers only."""
    sel = "|".join(str(i) for i in layer_idx)
    return rf".*audio_tower\.layers\.({sel})\.self_attn\.(q_proj|k_proj|v_proj|out_proj)$"


class _BadAudio(Exception):
    """Raised internally when one utterance cannot be decoded."""


# Errors we treat as "this file is bad". Deliberately narrow: a bug, an OOM, or a KeyboardInterrupt
# must propagate rather than be absorbed into the drop budget. soundfile raises its own
# LibsndfileError (a subclass of RuntimeError), so that one is named explicitly instead of catching
# bare RuntimeError.
_RECOVERABLE: tuple[type[BaseException], ...] = (OSError, EOFError, ValueError, _BadAudio)
try:  # soundfile >= 0.12
    _RECOVERABLE = _RECOVERABLE + (sf.LibsndfileError,)
except AttributeError:  # pragma: no cover — older soundfile
    pass


class CDSDDataset(Dataset):
    """Audio dataset with a bounded, per-utterance tolerance for unreadable items.

    A single bad file must not kill a multi-hour run, but an unbounded number of silent drops would
    let a run "succeed" on a fraction of the audited corpus.

    On failure the item is replaced by a different VALID utterance rather than a sentinel. That keeps
    every batch full and correctly sized — returning None or a zero-loss placeholder would either
    crash the Trainer (which inspects batches before training_step) or silently dilute the effective
    batch and bias the averaged loss. Failures are counted per unique utterance id, so one
    permanently-corrupt file consumes exactly one unit of the budget however many epochs revisit it.
    """

    MAX_SUBSTITUTIONS = 8

    def __init__(self, rows: list[dict], max_failures: int):
        self.rows = rows
        self.max_failures = max_failures
        self.failures: dict[str, str] = {}   # uid -> reason, deduplicated across epochs

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, rec: dict) -> dict:
        audio, sr = sf.read(rec["audio_filepath"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            # never silently reinterpret at the wrong rate — padding below assumes 16 kHz
            import torchaudio

            audio = (
                torchaudio.transforms.Resample(sr, SR)(torch.from_numpy(audio).unsqueeze(0))
                .squeeze(0)
                .numpy()
            )
        if audio.size == 0:
            raise _BadAudio("empty waveform")
        if not np.isfinite(audio).all():
            raise _BadAudio("non-finite samples (NaN/Inf)")
        # Qwen3ASREncoder requires the mel length to be a multiple of n_window*2 = 100 frames. The
        # mel hop is 10 ms, so padding the waveform to a whole second satisfies it.
        rem = len(audio) % SR
        if rem:
            audio = np.pad(audio, (0, SR - rem))
        return {"audio": audio, "text": rec["text"]}

    def _record_failure(self, uid: str, why: str) -> None:
        if uid not in self.failures:
            self.failures[uid] = why
        if len(self.failures) > self.max_failures:
            raise RuntimeError(
                f"aborting: {len(self.failures)} distinct unreadable utterances exceeds the allowed "
                f"{self.max_failures} "
                f"({100*self.max_failures/max(len(self.rows),1):.1f}% of the training split). "
                f"First 5: {list(self.failures.items())[:5]}"
            )

    def __getitem__(self, i: int) -> dict:
        for attempt in range(self.MAX_SUBSTITUTIONS):
            rec = self.rows[(i + attempt) % len(self.rows)]
            uid = str(rec.get("uid", i))
            try:
                return self._load(rec)
            except _RECOVERABLE as exc:
                # _record_failure may raise the hard abort; that must NOT be caught here, so it is
                # called outside the try body above via this explicit re-raise path.
                reason = f"{type(exc).__name__}: {exc}"
            self._record_failure(uid, reason)
        raise RuntimeError(
            f"{self.MAX_SUBSTITUTIONS} consecutive utterances starting at index {i} were unreadable "
            f"— aborting rather than training on a degraded corpus"
        )


def build_collate(processor):
    """Build supervised batches: audio prompt + transcript as the assistant turn, prompt masked out."""
    tok = getattr(processor, "tokenizer", None)

    def collate(batch):
        # The dataset substitutes a valid utterance for any unreadable one, so a batch is never
        # empty here and no None/placeholder path is needed.
        convs, texts = [], []
        for b in batch:
            convs.append([{"role": "user", "content": [{"type": "audio", "audio": b["audio"]}]}])
            texts.append(b["text"])

        prompts = processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        target_ids = [
            tok(t + tok.eos_token, add_special_tokens=False)["input_ids"] for t in texts
        ]
        max_t = max(len(t) for t in target_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        in_ids = prompts["input_ids"]
        attn = prompts.get("attention_mask", torch.ones_like(in_ids))
        B, P = in_ids.shape
        full = torch.full((B, P + max_t), pad_id, dtype=in_ids.dtype)
        full_attn = torch.zeros((B, P + max_t), dtype=attn.dtype)
        labels = torch.full((B, P + max_t), -100, dtype=torch.long)
        full[:, :P] = in_ids
        full_attn[:, :P] = attn
        for i, t in enumerate(target_ids):
            full[i, P:P + len(t)] = torch.tensor(t, dtype=in_ids.dtype)
            full_attn[i, P:P + len(t)] = 1
            labels[i, P:P + len(t)] = torch.tensor(t, dtype=torch.long)

        out = {k: v for k, v in prompts.items() if k not in ("input_ids", "attention_mask")}
        out["input_ids"] = full
        out["attention_mask"] = full_attn
        out["labels"] = labels
        return out

    return collate


REQUIRED_KEYS = ("audio_filepath", "text", "speaker_id", "split", "uid")
VALID_SPLITS = ("train", "val", "test")

# Documented composition of results/probe/cdsd_lora_manifest.jsonl, built by build_probe_dataset.py
# --subset 1h. Compose-checked at launch so a truncated or re-built manifest cannot slip through.
EXPECTED_TRAIN = 33617
EXPECTED_ROWS = 47687
EXPECTED_CORPUS_DIR = "CDSD"
EXPECTED_SUBSET = "1h"

# Documented composition per corpus, checked at launch so a truncated or wrong manifest cannot slip
# through. Adding a corpus means adding a row here — the gate is never simply disabled.
CORPUS_PRESETS = {
    # name: (speakers, rows, train_rows, corpus dir under data/processed, subset or None)
    "cdsd": (44, 47687, 33617, "CDSD", "1h"),
    "uaspeech": (15, 11402, 6075, "UASPEECH", "uaspeech"),
}
MAX_DROP_FRAC_CEILING = 0.05  # hard ceiling on --max-drop-frac, see main()
_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def load_and_validate(manifest: Path, min_train: int, preflight: int, tolerance: float,
                      exp_speakers: int = EXPECTED_SPEAKERS, exp_rows: int = EXPECTED_ROWS,
                      exp_train: int = EXPECTED_TRAIN, corpus_dir: str = EXPECTED_CORPUS_DIR,
                      subset: str | None = EXPECTED_SUBSET):
    """Load the manifest and hard-gate on schema, split integrity and corpus composition.

    The expected-composition constants are parameters rather than literals so a second corpus can be
    run through the same gates. Defaults remain the CDSD 1h values, so an omitted argument cannot
    silently weaken the check for the original study.
    """
    rows = []
    with open(manifest, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            miss = [k for k in REQUIRED_KEYS if not r.get(k)]
            if miss:
                raise SystemExit(f"{manifest}:{n} missing required field(s) {miss}")
            if r["split"] not in VALID_SPLITS:
                raise SystemExit(
                    f"{manifest}:{n} has unknown split {r['split']!r}; expected one of {VALID_SPLITS}. "
                    f"Rows with unknown splits would be counted in the compose check but silently "
                    f"excluded from training."
                )
            rows.append(r)
    if not rows:
        raise SystemExit(f"{manifest} is empty")

    by_split: dict[str, list[dict]] = {s: [] for s in VALID_SPLITS}
    for r in rows:
        by_split[r["split"]].append(r)
    train, val, test = (by_split[s] for s in VALID_SPLITS)
    for name, part in zip(VALID_SPLITS, (train, val, test)):
        if not part:
            raise SystemExit(f"split {name!r} is empty — refusing to train on an incomplete split")

    spk = {s: {r["speaker_id"] for r in v} for s, v in by_split.items()}
    leaks = {}
    for a in VALID_SPLITS:
        for b in VALID_SPLITS:
            if a < b and (spk[a] & spk[b]):
                leaks[f"{a}|{b}"] = sorted(spk[a] & spk[b])
    if leaks:
        raise SystemExit(f"SPEAKER LEAK across splits — refusing to train: {leaks}")

    # ---- corpus compose check: counts, speakers, and no unexpected corpora ----
    all_spk = {r["speaker_id"] for r in rows}
    if len(all_spk) != exp_speakers:
        raise SystemExit(
            f"compose-check failed: {len(all_spk)} speakers, expected {exp_speakers}"
        )
    def corpus_of(p: str) -> str:
        parts = Path(p).parts
        if "processed" in parts:
            i = parts.index("processed")
            if i + 1 < len(parts):
                return parts[i + 1]
        return "?"

    corpora = {corpus_of(r["audio_filepath"]) for r in rows}
    unexpected = {c for c in corpora if c != corpus_dir}
    if unexpected:
        raise SystemExit(f"compose-check failed: unexpected corpora in manifest: {sorted(unexpected)}")
    subsets = {r.get("subset") or Path(r["audio_filepath"]).parent.name for r in rows}
    if subset is not None and subsets != {subset}:
        raise SystemExit(
            f"compose-check failed: subsets {sorted(subsets)}, expected only {subset!r}. "
            f"The 10h subset carries the known alignment defect and must not be mixed in."
        )
    for label, got, want in (("rows", len(rows), exp_rows), ("train", len(train), exp_train)):
        if abs(got - want) > tolerance * want:
            raise SystemExit(
                f"compose-check failed: {label}={got}, documented {want} "
                f"(outside ±{tolerance:.0%}). Rebuild the manifest or pass --compose-tolerance."
            )
    if len(train) < min_train:
        raise SystemExit(f"train split has {len(train)} rows, below --min-train {min_train}")

    # preflight: header-only read catches missing/corrupt audio before a multi-hour run starts
    bad = []
    sample = train if preflight <= 0 or preflight >= len(train) else train[:: max(1, len(train) // preflight)]
    for r in sample:
        try:
            if sf.info(r["audio_filepath"]).frames == 0:
                bad.append(f"{r['uid']}: 0 frames")
        except _RECOVERABLE as exc:
            bad.append(f"{r['uid']}: {type(exc).__name__}: {exc}")
    if bad:
        raise SystemExit(
            f"preflight audio check failed on {len(bad)}/{len(sample)} sampled rows; first 5: {bad[:5]}"
        )

    audit = {
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "rows": len(rows), "expected_rows": exp_rows,
        "train": len(train), "expected_train": exp_train,
        "val": len(val), "test": len(test),
        "speakers_total": len(all_spk),
        "speakers_per_split": {k: sorted(v) for k, v in spk.items()},
        "utts_per_split": {k: len(v) for k, v in by_split.items()},
        "speaker_leaks": leaks,
        "corpora": sorted(corpora), "subsets": sorted(subsets),
        "preflight_checked": len(sample),
    }
    return train, val, test, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    help="all | lower | middle | upper | single_<1..24>")
    # Base, NOT the zh-aug fine-tune — see module docstring.
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_lora_manifest.jsonl")
    ap.add_argument("--out-root", type=Path, default=BASE / "models" / "layerprobe_lora")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42,
                    help="Training seed. Enters the run signature, so two seeds cannot collide on "
                         "one output directory. Default 42 reproduces the pre-2026-08-14 runs, which "
                         "used the HF Trainer default.")
    ap.add_argument("--lr", type=float, default=3e-5)
    # per-device 2: long dysarthric clips OOM at 4 on unified memory (playbook long-clip rule)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--effective-batch", type=int, default=128)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--min-train", type=int, default=-1,
                    help="hard gate: refuse to start on a truncated manifest. Default -1 derives it "
                         "as 90%% of the chosen --corpus preset's documented training rows.")
    ap.add_argument("--compose-tolerance", type=float, default=0.02,
                    help="allowed fractional deviation from documented row/train counts")
    ap.add_argument("--corpus", choices=tuple(CORPUS_PRESETS), default="cdsd",
                    help="expected-composition preset for the compose-check gate")
    ap.add_argument("--max-drop-frac", type=float, default=0.02,
                    help="abort if more than this fraction of training utterances is unreadable")
    ap.add_argument("--preflight", type=int, default=2000,
                    help="rows to header-check before training (0 = all)")
    ap.add_argument("--max-train", type=int, default=0, help="cap training rows (smoke tests)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest checkpoint in the run directory")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a run directory that already holds an adapter")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU execution (default: refuse — a CPU run would take weeks)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report trainable-parameter counts and exit without training")
    args = ap.parse_args()

    # Fail-fast argument validation. Each of these was reachable and degraded a gate rather than
    # erroring: --epochs 0 writes an adapter that passes the completion gate without a single
    # optimizer step; a non-finite --compose-tolerance makes the row-count check vacuous; a negative
    # --max-train silently trains on a different subset than the one audited.
    import math as _math
    if args.epochs <= 0:
        raise SystemExit(f"--epochs must be > 0, got {args.epochs}: a zero-epoch run writes adapter "
                         f"weights and passes the completion gate without training")
    if not _math.isfinite(args.compose_tolerance) or not (0 <= args.compose_tolerance < 1):
        raise SystemExit(f"--compose-tolerance must be finite and in [0, 1), got {args.compose_tolerance}")
    if args.max_train < 0:
        raise SystemExit(f"--max-train must be >= 0 (0 = no cap), got {args.max_train}")
    if not (0 <= args.seed < 2**32):
        raise SystemExit(f"--seed must be in [0, 2**32), got {args.seed}")


    idx = layers_for(args.variant)
    print(f"variant={args.variant}  encoder layers (0-based): {idx}")

    # Bound the drop budget. Without a ceiling a caller could pass e.g. 0.5 and still satisfy the
    # completion gate while half the training slots were deterministic substitutions rather than
    # distinct real utterances — dispersed bad rows never trip the consecutive-failure abort.
    if not 0.0 <= args.max_drop_frac <= MAX_DROP_FRAC_CEILING:
        raise SystemExit(
            f"--max-drop-frac {args.max_drop_frac} outside [0, {MAX_DROP_FRAC_CEILING}]. A higher "
            f"tolerance would let the completion gate pass on a substantially substituted corpus."
        )

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA unavailable — refusing to train on CPU. Pass --allow-cpu to override.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_spk, exp_rows, exp_train, corpus_dir, subset = CORPUS_PRESETS[args.corpus]
    # -1 is the sentinel meaning "derive from the preset". Any other non-positive value would make
    # `len(train) < min_train` vacuously false and silently disable the gate, so reject it.
    if args.min_train == -1:
        args.min_train = int(0.9 * exp_train)
    elif args.min_train <= 0:
        raise SystemExit(
            f"--min-train {args.min_train} would disable the minimum-train gate. Pass -1 to derive "
            f"it from the --corpus preset (90% of {exp_train}), or a positive row count."
        )
    train, val, test, audit = load_and_validate(
        args.manifest, args.min_train, args.preflight, args.compose_tolerance,
        exp_speakers=exp_spk, exp_rows=exp_rows, exp_train=exp_train,
        corpus_dir=corpus_dir, subset=subset,
    )
    if args.max_train:
        train = train[: args.max_train]
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"speakers={audit['speakers_total']}  (speaker leaks: {audit['speaker_leaks'] or 'none'})")

    if args.effective_batch % args.batch_size:
        raise SystemExit(
            f"--effective-batch {args.effective_batch} is not divisible by --batch-size "
            f"{args.batch_size}; the real effective batch would silently differ"
        )
    grad_acc = args.effective_batch // args.batch_size

    # Run identity covers every setting that changes the result, plus the manifest CONTENT hash and
    # the code hash — so different data or different hyperparameters can never share a directory.
    code_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + (Path(__file__).parent / "qwen3asr_compat.py").read_bytes()
    ).hexdigest()[:8]
    sig_payload = {
        "model": args.model, "layers": idx, "rank": args.rank, "alpha": args.alpha,
        "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
        "epochs": args.epochs, "eff_batch": args.effective_batch, "batch_size": args.batch_size,
        "max_train": args.max_train, "manifest_sha256": audit["manifest_sha256"],
        "code_sha256": code_hash, "seed": args.seed,
    }
    sig = hashlib.sha256(json.dumps(sig_payload, sort_keys=True).encode()).hexdigest()[:10]
    out_dir = args.out_root / f"{args.variant}__{sig}"
    adapter_dir = out_dir / "adapter"

    if adapter_dir.exists() and not (args.overwrite or args.resume):
        raise SystemExit(
            f"{adapter_dir} already exists — pass --resume to continue or --overwrite to replace"
        )
    if args.overwrite and out_dir.exists():
        # actually clear it: leaving stale checkpoints behind is how runs blend into each other
        import shutil

        shutil.rmtree(out_dir)
        print(f"--overwrite: cleared {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ML-05 free-disk gate. save_strategy="epoch" with save_total_limit=2 keeps two full checkpoints
    # plus the final adapter; on this model a checkpoint is ~5 GB. Refuse to start a multi-hour run
    # that will die on ENOSPC part-way and leave a half-written checkpoint tree behind.
    import shutil as _shutil
    import math as _m2
    _MIN_FREE_GB = float(os.environ.get("LAYERPROBE_MIN_FREE_GB", "25"))
    # A NaN or negative override would make the comparison below always false and silently
    # disable this gate — reject rather than skip.
    if not _m2.isfinite(_MIN_FREE_GB) or _MIN_FREE_GB < 0:
        raise SystemExit(f"LAYERPROBE_MIN_FREE_GB must be finite and >= 0, got {_MIN_FREE_GB}")
    _free_gb = _shutil.disk_usage(out_dir).free / 1e9
    if _free_gb < _MIN_FREE_GB:
        raise SystemExit(
            f"only {_free_gb:.1f} GB free at {out_dir} — need >= {_MIN_FREE_GB:.0f} GB for two "
            f"checkpoints plus the exported adapter. Free space or set LAYERPROBE_MIN_FREE_GB."
        )
    print(f"  disk free       : {_free_gb:.1f} GB (>= {_MIN_FREE_GB:.0f} GB required)")
    (out_dir / "_speaker_disjoint_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (out_dir / "_run_config.json").write_text(
        json.dumps({**sig_payload, "variant": args.variant, "run_sig": sig}, indent=2),
        encoding="utf-8",
    )

    # ---- Rule 4: explicit pre-launch statement of what this run is and how it will be judged ----
    print("\n--- pre-launch check ---")
    print(f"  hypothesis      : Probe B predicts the UPPER band (17-24) beats the proportional")
    print(f"                    analogue of Muthu's winner (9-16) on Qwen3-ASR.")
    print(f"  success criteria: test-speaker CER per variant, compared against variant 'all';")
    print(f"                    a band is 'competitive' within ~3% relative of all-layer adaptation.")
    print(f"  model           : {args.model} (base — zh-aug saw 82% of this data)")
    print(f"  data            : {len(train)} train / {len(val)} val / {len(test)} test, "
          f"{audit['speakers_total']} speakers, splits speaker-disjoint")
    print(f"  manifest sha256 : {audit['manifest_sha256'][:16]}…")
    print(f"  run dir         : {out_dir}")
    print(f"  drop tolerance  : {args.max_drop_frac:.1%} of train ({int(args.max_drop_frac*len(train))} utts)")
    print(f"  running signals : (1) loss every 25 steps, (2) per-epoch checkpoint, "
          f"(3) skipped-utterance counter")
    print("--- end pre-launch check ---\n")

    # Frozen base in bf16, trainable LoRA params upcast to fp32 below. Loading the whole model in
    # fp32 and relying on bf16 autocast breaks Qwen3-ASR: the audio tower emits bf16 under autocast
    # while inputs_embeds stays fp32, and masked_scatter in modeling_qwen3_asr rejects the mismatch.
    # Only LoRA parameters are optimised here, so fp32 master weights on those is what matters.
    # Seed EVERYTHING before the base model is loaded and before get_peft_model initialises the LoRA
    # A/B matrices. TrainingArguments(seed=...) is applied by the Trainer only at train() time, which
    # is AFTER adapter initialisation — so relying on it alone would give every seed identical
    # starting adapter weights, reducing a multi-seed study to a data-order study.
    from transformers import set_seed as _set_seed
    _set_seed(args.seed)
    print(f"  seeded (model init + data order): {args.seed}")

    model, cfg, report = load_qwen3_asr(args.model, dtype=torch.bfloat16, device=None)
    print(f"load OK — missing={len(report['missing'])} unexpected={len(report['unexpected'])}")

    from peft import LoraConfig, get_peft_model

    rx = lora_target_regex(idx)
    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=rx, task_type=None,
    )
    matched = [n for n, _ in model.named_modules() if re.match(rx, n)]
    if len(matched) != len(idx) * 4:
        raise SystemExit(f"LoRA regex matched {len(matched)} modules, expected {len(idx)*4}")
    if any("audio_tower" not in n for n in matched):
        raise SystemExit(f"LoRA regex matched non-encoder modules: "
                         f"{[n for n in matched if 'audio_tower' not in n][:5]}")
    model = get_peft_model(model, lcfg)

    # decoder stays frozen throughout, to isolate acoustic-phonetic adaptation (paper §III.E)
    n_lora = n_enc = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.data = p.data.to(torch.float32)  # fp32 master weights on the trained params
            p.requires_grad = True
            n_lora += p.numel()
            if "audio_tower" in name:
                n_enc += p.numel()
        else:
            p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable LoRA params: {n_lora:,} ({100*n_lora/total:.3f}% of {total:,})")
    if n_lora == 0:
        raise SystemExit("no LoRA parameters matched — check the target regex against module names")
    if n_lora != n_enc:
        raise SystemExit(
            f"{n_lora - n_enc:,} trainable params lie OUTSIDE the audio encoder — decoder must stay frozen"
        )

    if args.dry_run:
        print(f"matched modules: {len(matched)} (expect {len(idx)*4})")
        for n in matched[:8]:
            print("   ", n)
        print(f"run dir would be: {out_dir}")
        return

    from transformers import AutoProcessor, Trainer, TrainingArguments

    proc = AutoProcessor.from_pretrained(args.model)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,  # DGX GB10 unified memory deadlocks with workers > 0 under PEFT
        seed=args.seed,
        data_seed=args.seed,
    )

    torch.backends.cuda.enable_cudnn_sdp(False)  # cuDNN SDP segfaults Qwen3-ASR attention

    max_failures = int(args.max_drop_frac * len(train))
    ds = CDSDDataset(train, max_failures=max_failures)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=build_collate(proc),
    )

    resume = False
    if args.resume:
        # auto-inventory: always continue from the DEEPEST checkpoint present, and fail rather than
        # silently converting an explicit resume into a fresh ~5 h run
        # strictly checkpoint-<int>: a name like checkpoint-malformed-500 must not be selectable
        dirs = [p for p in out_dir.glob("checkpoint-*") if p.is_dir()]
        ckpts = sorted(
            (p for p in dirs if _CKPT_RE.match(p.name)),
            key=lambda p: int(_CKPT_RE.match(p.name).group(1)),
        )
        malformed = [p.name for p in dirs if not _CKPT_RE.match(p.name)]
        if malformed:
            print(f"  ignoring malformed checkpoint dirs: {malformed}")
        if not ckpts:
            raise SystemExit(
                f"--resume given but no checkpoint-* found under {out_dir}. Drop --resume to start "
                f"fresh; failing here so a resume request never silently restarts a 5 h run."
            )
        resume = str(ckpts[-1])
        print(f"resume: {resume} (deepest of {len(ckpts)})")

    t0 = time.time()
    # ML-04 enforced runtime gate. Run ONE real forward+backward before committing to hours of
    # training, so a shape/dtype/OOM fault surfaces in seconds rather than after the first epoch.
    # This is a smoke test, not a training step: gradients are discarded and the model is restored
    # to eval-free zero-grad state before Trainer.train() starts from scratch.
    if not resume:
        _smoke_batch = trainer.get_train_dataloader()
        _b = next(iter(_smoke_batch))
        _b = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in _b.items()}
        model.train()
        # The Trainer runs forward under bf16 autocast (bf16=True). Calling model(**batch)
        # raw passes fp32 features into the bf16 audio tower and dies in conv2d, so the smoke
        # step MUST use the same context the real run will use or it tests a different graph.
        _amp = torch.autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"),
                              dtype=torch.bfloat16)
        with _amp:
            _out = model(**_b)
        if _out.loss is None or not torch.isfinite(_out.loss):
            raise SystemExit(f"smoke step produced non-finite loss ({_out.loss}) — aborting before "
                             f"the full run")
        _out.loss.backward()
        _n_grad = sum(1 for p_ in model.parameters() if p_.requires_grad and p_.grad is not None)
        if _n_grad == 0:
            raise SystemExit("smoke step produced no gradients on any trainable parameter — the "
                             "LoRA adapter is not in the graph; aborting")
        model.zero_grad(set_to_none=True)
        print(f"  smoke step OK   : loss={_out.loss.item():.4f}, {_n_grad} trainable tensors have grads")
        del _out, _b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(adapter_dir))

    # ---- completion gate: an adapter must exist AND the run must have consumed the corpus ----
    written = list(adapter_dir.glob("*"))
    if not any(f.suffix in (".safetensors", ".bin") for f in written):
        raise SystemExit(f"no adapter weights written to {adapter_dir} — files: {[f.name for f in written]}")

    n_dropped = len(ds.failures)  # distinct utterance ids, not access attempts
    completion = 1.0 - n_dropped / max(len(train), 1)
    if ds.failures:
        (out_dir / "_skipped_utterances.txt").write_text(
            "\n".join(f"{uid}: {why}" for uid, why in sorted(ds.failures.items())), encoding="utf-8"
        )
    print(f"completion: {completion:.2%} of the training split readable "
          f"({n_dropped} skipped, tolerance {max_failures})")
    if n_dropped > max_failures:
        raise SystemExit(
            f"completion gate failed: {n_dropped} skipped exceeds {max_failures} "
            f"({args.max_drop_frac:.1%}); adapter at {adapter_dir} is NOT trustworthy"
        )

    (out_dir / "_run_result.json").write_text(
        json.dumps({"variant": args.variant, "run_sig": sig, "hours": round((time.time()-t0)/3600, 3),
                    "train_rows": len(train), "skipped": n_dropped,
                    "completion": round(completion, 5),
                    "adapter_files": sorted(f.name for f in written)}, indent=2),
        encoding="utf-8",
    )
    print(f"\n{args.variant}: trained in {(time.time()-t0)/3600:.2f} h -> {adapter_dir}")
    print(f"  files: {[f.name for f in written]}")


if __name__ == "__main__":
    main()
