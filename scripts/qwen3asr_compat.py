#!/usr/bin/env python3
"""Loader shims for Qwen3-ASR checkpoints under transformers >= 5.14.

The Qwen3-ASR checkpoints (base `Qwen/Qwen3-ASR-1.7B`, last touched 2026-01-30, and every CANDOR
fine-tune derived from it) store weights under the pre-restructure `thinker.*` namespace. transformers
5.14's `Qwen3ASRForConditionalGeneration` expects `model.*` and ships no
`_checkpoint_conversion_mapping`, so a plain `from_pretrained` silently loads a RANDOMLY INITIALISED
model: measured overlap between checkpoint keys and model parameters is exactly 0/710, and the only
signal is a `MISSING`/`UNEXPECTED` warning block that is easy to scroll past.

`load_qwen3_asr` below does the remap and asserts the result, so a silent random-init cannot happen.
Anything probing or fine-tuning a Qwen3-ASR checkpoint must go through it.

Requires venv_qwen3asr on DGX (torch 2.12.1+cu130, transformers 5.14.1) — the main venv pins
transformers 4.57.6, which does not know the `qwen3_asr` model type at all.
"""
from __future__ import annotations

import torch

# checkpoint prefix -> model prefix, longest-first so the proj1/proj2 renames win over the generic
# audio_tower rule
_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
    ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
    ("thinker.audio_tower.", "model.audio_tower."),
    ("thinker.multi_modal_projector.", "model.multi_modal_projector."),
    ("thinker.lm_head.", "lm_head."),
    ("thinker.model.", "model.language_model."),
)

# recomputed at init rather than restored; absence from the checkpoint is expected
_DERIVED_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rotary_emb.original_inv_freq",
    "positional_embedding.positional_embedding",
)


def remap_key(key: str) -> str:
    for src, dst in _KEY_MAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    return key


def _is_derived(name: str) -> bool:
    return any(name.endswith(s) for s in _DERIVED_SUFFIXES)


def cfg_get(cfg, *path, default=None):
    """Walk a config path tolerating dict-vs-object nesting.

    transformers 5.14 returns `thinker_config` as a plain dict for some load paths and as a config
    object for others, so `cfg.thinker_config.audio_config` is not safe on its own.
    """
    node = cfg
    for key in path:
        if node is None:
            return default
        node = node.get(key, None) if isinstance(node, dict) else getattr(node, key, None)
    return default if node is None else node


def build_config(model_id: str):
    """Build a Qwen3ASRConfig from the checkpoint's nested `thinker_config`.

    `AutoConfig.from_pretrained` on these checkpoints returns a config whose TOP-LEVEL
    `audio_config`/`text_config` are library DEFAULTS — the real values sit unused under
    `thinker_config`. Most defaults happen to coincide with the checkpoint (d_model 1024, 24 audio
    layers, text hidden 2048, 28 text layers), which makes the bug easy to miss, but
    `audio_config.output_dim` defaults to 3584 against the checkpoint's 2048. Building the model from
    that config yields a multi_modal_projector of the wrong width.
    """
    import json
    import os

    from huggingface_hub import hf_hub_download
    from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASRConfig

    path = (
        os.path.join(model_id, "config.json")
        if os.path.isdir(model_id)
        else hf_hub_download(model_id, "config.json")
    )
    raw = json.load(open(path, encoding="utf-8"))
    thinker = raw.get("thinker_config")
    if thinker is None:  # already in the flat post-restructure layout
        return Qwen3ASRConfig(**{k: v for k, v in raw.items() if k != "model_type"})

    passthrough = {
        k: thinker[k]
        for k in ("audio_token_id", "timestamp_token_id", "pad_token_id", "eos_token_id",
                  "initializer_range", "tie_word_embeddings", "token_classification_bias")
        if k in thinker
    }

    # the checkpoint's sub-config model_type strings predate the 5.14 rename and are not in
    # CONFIG_MAPPING: audio "qwen3_asr_audio_encoder" -> "qwen3_asr_encoder", text "qwen3_asr_text"
    # -> "qwen3"
    audio_cfg = dict(thinker["audio_config"])
    audio_cfg["model_type"] = "qwen3_asr_encoder"
    text_cfg = dict(thinker["text_config"])
    if text_cfg.get("model_type") not in (None, "qwen3"):
        text_cfg["model_type"] = "qwen3"

    return Qwen3ASRConfig(
        audio_config=audio_cfg,
        text_config=text_cfg,
        architectures=raw.get("architectures"),
        dtype=thinker.get("dtype") or raw.get("dtype"),
        **passthrough,
    )


def load_qwen3_asr(model_id: str, dtype=torch.float32, device=None, max_unmatched: int = 0):
    """Load a Qwen3-ASR checkpoint with the thinker.* -> model.* remap applied.

    Raises RuntimeError if more than `max_unmatched` non-derived parameters fail to receive checkpoint
    weights, so a silently random-initialised encoder is impossible.
    """
    from transformers import Qwen3ASRForConditionalGeneration

    cfg = build_config(model_id)
    state = _read_state_dict(model_id)
    # remap in place and drop the source dict as we go: holding both copies of a 1.7B checkpoint
    # doubles peak RAM for no reason
    remapped: dict = {}
    origin: dict[str, str] = {}
    for k in list(state):
        nk = remap_key(k)
        # Two distinct source keys can map onto the same target (e.g. a checkpoint carrying both the
        # legacy thinker.* form and an already-flat model.* form of one tensor). A plain assignment
        # would silently drop one, and the later missing/unexpected key-set comparison compares sets
        # only, so it could not detect the loss. Fail closed instead.
        if nk in remapped:
            raise RuntimeError(
                f"{model_id}: checkpoint keys {origin[nk]!r} and {k!r} both remap to {nk!r}; "
                f"one tensor would be silently discarded"
            )
        origin[nk] = k
        remapped[nk] = state.pop(k).to(dtype)
    del state, origin

    # transformers >= 5.14 refuses `state_dict=` alongside a model id, so build from config and load
    # the remapped weights explicitly.
    model = Qwen3ASRForConditionalGeneration(cfg)
    model = model.to(dtype)

    target = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    missing = sorted(
        n for n in target - set(remapped) if not _is_derived(n) and n != "lm_head.weight"
    )
    unexpected = sorted(set(remapped) - target)

    if len(missing) > max_unmatched:
        raise RuntimeError(
            f"{model_id}: {len(missing)} parameters would be randomly initialised "
            f"(first 10: {missing[:10]}). Refusing to load — the remap in qwen3asr_compat is stale."
        )

    # Unexpected keys mean checkpoint tensors the model never consumed. `lm_head.weight` is expected
    # (tied to the input embedding, and present in the base repo but not the fine-tunes). Anything else
    # means the remap is stale and real weights are being silently discarded — fail rather than train on
    # a partially-loaded model.
    unexpected_unknown = [n for n in unexpected if n != "lm_head.weight"]
    if unexpected_unknown:
        raise RuntimeError(
            f"{model_id}: {len(unexpected_unknown)} checkpoint tensors would be silently discarded "
            f"(first 10: {unexpected_unknown[:10]}). The remap in qwen3asr_compat is stale."
        )

    incompatible = model.load_state_dict(remapped, strict=False)
    model.tie_weights()

    # Fine-tuned checkpoints omit lm_head.weight when it is tied to the input embedding (707 keys vs
    # the base repo's 708). tie_weights() restores it, so it is only a real failure if the tie did not
    # actually take effect.
    tied_ok = False
    if "lm_head.weight" in incompatible.missing_keys:
        head = getattr(model, "lm_head", None)
        emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        tied_ok = (
            head is not None
            and emb is not None
            and head.weight.data_ptr() == emb.weight.data_ptr()
        )

    still_missing = [
        n
        for n in incompatible.missing_keys
        if not _is_derived(n) and not (n == "lm_head.weight" and tied_ok)
    ]
    if len(still_missing) > max_unmatched:
        raise RuntimeError(
            f"{model_id}: load_state_dict left {len(still_missing)} params uninitialised "
            f"(first 10: {still_missing[:10]})."
        )

    if device is not None:
        model = model.to(device)
    return model.eval(), cfg, {"missing": missing, "unexpected": unexpected}


def _read_state_dict(model_id: str) -> dict:
    """Read a full state dict from a local dir or HF-cached repo, sharded or single-file."""
    import glob
    import json
    import os

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    root = model_id
    if not os.path.isdir(root):
        root = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])

    index = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index):
        shards = sorted({v for v in json.load(open(index))["weight_map"].values()})
        files = [os.path.join(root, s) for s in shards]
    else:
        files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {root}")

    state: dict = {}
    for f in files:
        shard = load_file(f)
        # a key present in two shards means one tensor would silently clobber the other, and the
        # missing/unexpected key-set comparison downstream cannot detect it
        clash = set(shard) & set(state)
        if clash:
            raise RuntimeError(
                f"duplicate tensor keys across shards in {root} (e.g. {sorted(clash)[:5]}) — "
                f"refusing to load a checkpoint whose shards overlap"
            )
        state.update(shard)
    return state


def encoder_of(model):
    """Return the 24-layer Qwen3ASREncoder audio tower."""
    return model.model.audio_tower


def make_encoder_inputs(processor, audio, sampling_rate: int = 16000, device=None):
    """Build (input_features, input_features_mask) for Qwen3ASREncoder.forward.

    The encoder requires padded_feature_length to be a multiple of n_window*2 and takes an explicit
    mask, so the Whisper-style feature extractor output cannot be passed through unmodified.
    """
    fe = getattr(processor, "feature_extractor", processor)
    inputs = fe(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
    )
    feats = inputs["input_features"]
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones(feats.shape[0], feats.shape[-1], dtype=torch.long)
    if device is not None:
        feats, mask = feats.to(device), mask.to(device)
    return feats, mask
