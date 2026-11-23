#!/usr/bin/env python3
"""Extract per-layer Qwen3-ASR encoder representations for the probing study.

Writes one .npz shard per speaker containing, for all 25 taps (index 0 = post-conv encoder input,
1..24 = transformer block outputs):

    frames   (25, sum_T, 1024) float16   layer-major so a probe can read one layer contiguously
    offsets  (n_utt + 1,)      int32     frame span of each utterance within `frames`
    pooled   (n_utt, 25, 1024) float16   mean over valid frames; feeds Probe C and the similarity arm
    utt_ids  (n_utt,)          str

Frames are kept because Probe B is a sequence-level CTC probe. Probe C and the cross-condition
similarity analysis need only `pooled`, so `--pooled-only` skips the bulk of the storage.

MUST be run through qwen3asr_compat.load_qwen3_asr — a plain from_pretrained silently yields a
randomly-initialised model on these checkpoints (0/710 key overlap). See that module's docstring.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/extract_layer_features.py \
        --manifest results/probe/cdsd_probe_manifest.jsonl \
        --model Qwen/Qwen3-ASR-1.7B --tag base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3asr_compat import encoder_of, load_qwen3_asr, make_encoder_inputs  # noqa: E402

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
SR = 16000


def read_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torchaudio

        audio = (
            torchaudio.transforms.Resample(sr, SR)(torch.from_numpy(audio).unsqueeze(0))
            .squeeze(0)
            .numpy()
        )
    return audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--tag", required=True, help="output subdirectory name, e.g. base / zh_aug")
    ap.add_argument("--out-root", type=Path, default=BASE / "results" / "probe" / "features")
    ap.add_argument("--pooled-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N utterances")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker_id"]].append(r)

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} ...")
    # Load on CPU and move ONLY the audio tower to the accelerator. The full model is ~1.7B params
    # (~7 GB fp32) but the encoder is ~300M; the DGX GB10's memory is unified and shared with whatever
    # else is running, so moving the whole model first is what OOMs.
    model, cfg, report = load_qwen3_asr(args.model, dtype=torch.float32, device=None)
    print(f"  load OK — missing={len(report['missing'])} (0 required)")
    enc = encoder_of(model)
    try:
        del model.model.language_model
        del model.lm_head
    except AttributeError:
        pass
    enc = enc.to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        n_par = sum(p.numel() for p in enc.parameters())
        print(f"  encoder on {args.device}: {n_par/1e6:.0f}M params, "
              f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)

    n_done = 0
    t0 = time.time()
    for spk in sorted(by_speaker):
        shard = out_dir / f"{spk}.npz"
        if shard.exists():
            print(f"  {spk}: exists, skipping")
            n_done += len(by_speaker[spk])
            continue

        utts = by_speaker[spk]
        frame_chunks: list[np.ndarray] = []
        pooled_rows: list[np.ndarray] = []
        offsets = [0]
        kept_ids = []

        for rec in utts:
            try:
                audio = read_audio(rec["audio_filepath"])
            except Exception as exc:  # noqa: BLE001
                print(f"    !! {rec['utt_id']}: {type(exc).__name__}: {exc}")
                continue
            feats, mask = make_encoder_inputs(proc, audio, SR, device=args.device)
            with torch.no_grad():
                out = enc(feats, mask, output_hidden_states=True)
            hs = out.hidden_states  # 25 x (T, D) or (1, T, D)
            stacked = torch.stack([h[0] if h.dim() == 3 else h for h in hs], dim=0)  # (25, T, D)
            pooled_rows.append(stacked.mean(dim=1).to(torch.float16).cpu().numpy())
            if not args.pooled_only:
                frame_chunks.append(stacked.to(torch.float16).cpu().numpy())
                offsets.append(offsets[-1] + stacked.shape[1])
            kept_ids.append(rec["utt_id"])
            n_done += 1

        if not kept_ids:
            print(f"  {spk}: nothing usable")
            continue

        payload = {
            "pooled": np.stack(pooled_rows, axis=0),
            "utt_ids": np.array(kept_ids),
        }
        if not args.pooled_only:
            payload["frames"] = np.concatenate(frame_chunks, axis=1)  # (25, sum_T, D)
            payload["offsets"] = np.array(offsets, dtype=np.int32)
        np.savez(shard, **payload)

        rate = n_done / max(time.time() - t0, 1e-6)
        size_mb = shard.stat().st_size / 1e6
        print(f"  {spk}: {len(kept_ids):4d} utts -> {shard.name} ({size_mb:7.1f} MB)  "
              f"[{n_done}/{len(rows)}, {rate:.1f} utt/s]")

    print(f"\ndone: {n_done} utterances in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
