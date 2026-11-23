#!/usr/bin/env python3
"""Per-layer Whisper-small encoder features, in the SAME shard format as the Qwen extractor.

Writes one .npz per speaker with:
    frames   (n_layers, sum_T, D) float16   -- concatenated per-utterance frame sequences
    offsets  (n_utts + 1,) int32            -- utterance boundaries into `frames`
    pooled   (n_utts, n_layers, D) float16  -- mean-pooled per utterance
    utt_ids  (n_utts,) str
so that train_probe_b_ctc.py and train_probe_c_difficulty.py consume it unchanged.

THE CENTRAL HAZARD, and the reason this is not a two-line edit of the Qwen extractor.
Whisper's feature extractor ALWAYS pads the waveform to 30 s: `input_features` is (80, 3000) for every
utterance, and the encoder emits a fixed 1500 frames at 20 ms regardless of how much real audio there
was. CDSD utterances average ~2.4 s, i.e. ~120 real frames. Storing the raw 1500 would:
  (a) inflate the feature set to roughly half a terabyte, and
  (b) far worse, train the phoneme probe on ~92% padding frames -- which fails silently, producing a
      PER curve that describes Whisper's treatment of silence rather than of speech.
So each utterance is sliced to ceil(duration / 0.02) frames before anything is stored, and the script
asserts the slice is a real reduction rather than trusting it.

Layer indexing: `output_hidden_states=True` on a 12-block encoder yields 13 taps (embedding output +
12 blocks), the analogue of Qwen's 25 for 24 blocks.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/extract_layer_features_whisper.py --tag whisper_base
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"
SR = 16000
FRAME_S = 0.02          # whisper encoder output stride: 10 ms mel hop, x2 conv downsample
MAX_FRAMES = 1500       # 30 s / 0.02 -- the fixed padded length
N_TAPS_EXPECTED = 13    # embedding + 12 blocks


def read_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import torchaudio
        audio = (torchaudio.transforms.Resample(sr, SR)(torch.from_numpy(audio).unsqueeze(0))
                 .squeeze(0).numpy())
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("empty or non-finite waveform")
    return audio


def real_frames(n_samples: int) -> int:
    """Encoder frames containing actual audio, before Whisper's 30 s padding."""
    n = int(np.ceil((n_samples / SR) / FRAME_S))
    return max(1, min(n, MAX_FRAMES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_probe_manifest.jsonl")
    ap.add_argument("--tag", default="whisper_base")
    ap.add_argument("--out-root", type=Path, default=PROBE_DIR / "features")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pooled-only", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.manifest}")
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[r["speaker_id"]].append(r)
    print(f"{len(rows)} utts / {len(by_speaker)} speakers from {args.manifest.name}")

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
    enc = model.model.encoder.to(args.device).eval()
    n_blocks = len(enc.layers)
    print(f"encoder blocks: {n_blocks}  -> expecting {n_blocks + 1} taps")
    if n_blocks + 1 != N_TAPS_EXPECTED:
        raise SystemExit(f"expected {N_TAPS_EXPECTED} taps for whisper-small, model has {n_blocks + 1}")

    n_done, t0 = 0, time.time()
    slice_stats: list[tuple[int, int]] = []
    for spk in sorted(by_speaker):
        shard = out_dir / f"{spk}.npz"
        if shard.exists():
            print(f"  {spk}: shard exists, skipping")
            continue
        frame_chunks, pooled_rows, offsets, kept_ids = [], [], [0], []

        for rec in by_speaker[spk]:
            try:
                audio = read_audio(rec["audio_filepath"])
            except Exception as exc:  # noqa: BLE001
                print(f"    !! {rec['utt_id']}: {type(exc).__name__}: {exc}")
                continue
            keep = real_frames(len(audio))
            feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features
            feats = feats.to(args.device, dtype=torch.float32)
            with torch.no_grad():
                out = enc(feats, output_hidden_states=True)
            hs = out.hidden_states
            if len(hs) != N_TAPS_EXPECTED:
                raise SystemExit(f"got {len(hs)} taps, expected {N_TAPS_EXPECTED}")
            stacked = torch.stack([h[0] if h.dim() == 3 else h for h in hs], dim=0)  # (13, 1500, D)
            if stacked.shape[1] != MAX_FRAMES:
                raise SystemExit(f"expected {MAX_FRAMES} padded frames, got {stacked.shape[1]} — "
                                 f"the 30 s padding assumption no longer holds, and the slice below "
                                 f"would be wrong")
            slice_stats.append((keep, stacked.shape[1]))
            stacked = stacked[:, :keep, :]          # <-- drop the 30 s padding, see module docstring
            pooled_rows.append(stacked.mean(dim=1).to(torch.float16).cpu().numpy())
            if not args.pooled_only:
                frame_chunks.append(stacked.to(torch.float16).cpu().numpy())
                offsets.append(offsets[-1] + stacked.shape[1])
            kept_ids.append(rec["utt_id"])
            n_done += 1

        if not kept_ids:
            print(f"  {spk}: nothing usable")
            continue
        payload = {"pooled": np.stack(pooled_rows, axis=0), "utt_ids": np.array(kept_ids)}
        if not args.pooled_only:
            payload["frames"] = np.concatenate(frame_chunks, axis=1)
            payload["offsets"] = np.array(offsets, dtype=np.int32)
        np.savez(shard, **payload)
        print(f"  {spk}: {len(kept_ids):4d} utts -> {shard.name} "
              f"({shard.stat().st_size/1e6:7.1f} MB)  [{n_done}/{len(rows)}, "
              f"{n_done/max(time.time()-t0,1e-6):.1f} utt/s]")

    if slice_stats:
        kept = np.array([k for k, _ in slice_stats])
        print(f"\npadding slice: kept {kept.mean():.0f} of {MAX_FRAMES} frames on average "
              f"(min {kept.min()}, max {kept.max()}) — "
              f"{100*(1-kept.mean()/MAX_FRAMES):.1f}% of the padded tensor discarded")
        if kept.mean() > 0.5 * MAX_FRAMES:
            print("  WARNING: retained more than half the padded length; check the duration field")
    print(f"done: {n_done} utterances in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
