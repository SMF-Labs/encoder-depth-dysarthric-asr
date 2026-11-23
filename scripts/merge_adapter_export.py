#!/usr/bin/env python3
"""Merge a Phase-2 LoRA adapter into the base weights and export a qwen_asr-loadable checkpoint.

Why this exists. The adapters are PEFT modules over a transformers-5.14 `Qwen3ASRForConditionalGeneration`,
but the only decode path we trust is Qwen's own `qwen_asr.Qwen3ASRModel.transcribe`, which lives in the
MAIN venv (transformers 4.57.6) and expects the original `thinker.*` checkpoint namespace. Measured on
369 utterances that overlap the recorded Track 1 fair-eval predictions:

    recorded zh_ep2_fair decode ............ 41.87% CER
    transformers apply_chat_template+generate 124.57% CER   <-- do not use
    qwen_asr Qwen3ASRModel.transcribe ....... 32.38% CER

So scoring goes through qwen_asr, which means the merged model must be written back in `thinker.*` form.
This script does that: load base via qwen3asr_compat, attach the adapter, merge, invert the key remap,
and copy the tokenizer/processor/config files the loader needs.

The exported checkpoint keeps the ORIGINAL config.json from the base repo (nested thinker_config), which
is what qwen_asr expects — not the flattened config qwen3asr_compat builds for transformers 5.14.

Usage (DGX, venv_qwen3asr for the merge; the OUTPUT is then decoded in the main venv):
    python scripts/layerprobe/merge_adapter_export.py \
        --adapter models/layerprobe_lora/upper__734ee1d612/adapter \
        --out models/layerprobe_lora/upper__734ee1d612/merged
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3asr_compat import _KEY_MAP, load_qwen3_asr  # noqa: E402

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))

# files qwen_asr needs alongside the weights; copied verbatim from the base snapshot
_AUX_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
    "chat_template.json", "added_tokens.json", "special_tokens_map.json",
)


def invert_key(name: str) -> str:
    """model.* -> thinker.* — the inverse of qwen3asr_compat.remap_key.

    Longest destination first so `model.multi_modal_projector.linear_1.` is matched before the more
    general `model.` prefixes and cannot be mis-attributed.
    """
    for src, dst in sorted(_KEY_MAP, key=lambda kv: -len(kv[1])):
        if name.startswith(dst):
            return src + name[len(dst):]
    return name


def base_snapshot_dir(model_id: str) -> Path:
    if os.path.isdir(model_id):
        return Path(model_id)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.txt"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"{args.adapter} does not look like a PEFT adapter dir")

    dtype = getattr(torch, args.dtype)
    print(f"loading base {args.model} ...")
    model, cfg, report = load_qwen3_asr(args.model, dtype=dtype, device=None)
    print(f"  load OK — missing={len(report['missing'])}")

    from peft import PeftModel

    print(f"attaching adapter {args.adapter} ...")
    peft_model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    merged = peft_model.merge_and_unload()
    print("  merged")

    # sanity: the merge must actually have changed encoder weights, otherwise we would silently
    # export a copy of the base model and score it as if it were adapted
    tgt = json.loads((args.adapter / "adapter_config.json").read_text())["target_modules"]
    base_sd = {k: v for k, v in load_qwen3_asr(args.model, dtype=dtype, device=None)[0]
               .state_dict().items()}
    merged_sd = merged.state_dict()
    changed = sum(
        1 for k, v in merged_sd.items()
        if k in base_sd and v.shape == base_sd[k].shape and not torch.equal(v, base_sd[k])
    )
    print(f"  weights differing from base: {changed}")
    if changed == 0:
        raise SystemExit(
            f"merge produced a model identical to base (target_modules={tgt}) — refusing to export a "
            f"checkpoint that would be scored as if adapted"
        )
    del base_sd

    # .clone() is required, not just .contiguous(): lm_head.weight is tied to embed_tokens and shares
    # storage, which safetensors refuses to serialise. The base repo ships both keys, so cloning keeps
    # the exported key set identical to the checkpoint qwen_asr expects.
    out_sd = {}
    for k, v in merged_sd.items():
        out_sd[invert_key(k)] = v.detach().clone().contiguous().to(dtype)

    args.out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    weights = args.out / "model.safetensors"
    save_file(out_sd, str(weights), metadata={"format": "pt"})
    print(f"wrote {weights} ({weights.stat().st_size/1e9:.2f} GB, {len(out_sd)} tensors)")

    snap = base_snapshot_dir(args.model)
    copied = []
    for name in _AUX_FILES:
        src = snap / name
        if src.exists():
            shutil.copy2(src, args.out / name)
            copied.append(name)
    print(f"copied aux files: {copied}")

    (args.out / "_merge_provenance.json").write_text(
        json.dumps({"base": args.model, "adapter": str(args.adapter),
                    "target_modules": tgt, "tensors": len(out_sd),
                    "weights_changed_vs_base": changed, "dtype": args.dtype}, indent=2),
        encoding="utf-8",
    )
    print(f"\nexported -> {args.out}\n  decode this with qwen_asr from the MAIN venv")


if __name__ == "__main__":
    main()
