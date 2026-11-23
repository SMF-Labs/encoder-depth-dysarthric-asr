#!/usr/bin/env python3
"""Matched-pair feature extraction to test whether padding explains Muthu et al.'s PER plateau.

THE QUESTION. Their probe setup states input sequences "truncated to 1500 frames" -- Whisper's full
30 s padded window. Ours slices to the ~129 real frames per utterance. Their dysarthric PER runs
~80% -> ~65% across the stack and plateaus after layer 6; ours runs 81.79 -> 46.00 with its minimum at
layer 10. If feeding the CTC probe ~90% padding is what flattens the curve and inflates PER, then the
two results are methodologically different rather than contradictory, and our sliced measurement is the
cleaner one.

THE DESIGN. One extraction pass writes TWO shard sets over the SAME utterances and the SAME layers:
    whisper_pad1500        -- full 1500-frame sequences, as their setup describes
    whisper_slice_matched  -- sliced to real frames, our setup
Padding handling is therefore the ONLY difference between the two probe runs. Comparing our earlier
8,800-utterance sliced run against a new padded run would confound padding with sample size; this does
not.

Reduced scope so it fits alongside other work on the box: 7 taps (0,2,4,6,8,10,12) and ~1200
utterances. That is enough to resolve curve SHAPE, which is the claim at issue. Absolute PER will be
higher than the full run for both conditions -- they remain comparable to each other, which is the point.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/padding_replicate_extract.py --per-speaker 27
"""
from __future__ import annotations

import argparse
import json
import math
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
FRAME_S = 0.02
MAX_FRAMES = 1500
TAPS = [0, 2, 4, 6, 8, 10, 12]


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
    return max(1, min(int(np.ceil((n_samples / SR) / FRAME_S)), MAX_FRAMES))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-small")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_probe_manifest.jsonl")
    ap.add_argument("--per-speaker", type=int, default=27)
    ap.add_argument("--out-root", type=Path, default=PROBE_DIR / "features")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    args = ap.parse_args()

    if args.per_speaker <= 0:
        raise SystemExit(f"--per-speaker must be > 0, got {args.per_speaker}")
    if not torch.cuda.is_available() and not os.environ.get("LAYERPROBE_ALLOW_CPU"):
        raise SystemExit("no CUDA device — refusing to run on CPU (set LAYERPROBE_ALLOW_CPU=1)")
    import shutil as _sh
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        raise SystemExit(f"--min-free-gb must be finite and >= 0, got {args.min_free_gb}")
    free_gb = _sh.disk_usage(args.out_root.parent).free / 1e9
    if free_gb < args.min_free_gb:
        raise SystemExit(f"only {free_gb:.1f} GB free — need >= {args.min_free_gb:.0f} GB")
    print(f"  disk free: {free_gb:.1f} GB")

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_spk[r["speaker_id"]].append(r)
    sel: list[dict] = []
    for spk in sorted(by_spk):
        sel.extend(by_spk[spk][: args.per_speaker])
    print(f"  selected {len(sel)} utts from {len(by_spk)} speakers "
          f"({args.per_speaker}/speaker), taps {TAPS}")

    pad_dir = args.out_root / "whisper_pad1500"
    cut_dir = args.out_root / "whisper_slice_matched"
    for d in (pad_dir, cut_dir):
        d.mkdir(parents=True, exist_ok=True)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(args.model)
    enc = (WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)
           .model.encoder.to("cuda").eval())

    by_sel: dict[str, list[dict]] = defaultdict(list)
    for r in sel:
        by_sel[r["speaker_id"]].append(r)

    n_done, t0, keeps = 0, time.time(), []
    for spk in sorted(by_sel):
        pad_chunks, cut_chunks = [], []
        pad_off, cut_off, ids = [0], [0], []
        pad_pool, cut_pool = [], []
        for rec in by_sel[spk]:
            try:
                audio = read_audio(rec["audio_filepath"])
            except Exception as exc:  # noqa: BLE001
                print(f"    !! {rec['utt_id']}: {type(exc).__name__}: {exc}")
                continue
            keep = real_frames(len(audio))
            feats = proc(audio, sampling_rate=SR, return_tensors="pt").input_features.to(
                "cuda", dtype=torch.float32)
            with torch.no_grad():
                hs = enc(feats, output_hidden_states=True).hidden_states
            st = torch.stack([hs[i][0] if hs[i].dim() == 3 else hs[i] for i in TAPS], dim=0)
            if st.shape[1] != MAX_FRAMES:
                raise SystemExit(f"expected {MAX_FRAMES} padded frames, got {st.shape[1]}")
            pad_np = st.to(torch.float16).cpu().numpy()
            cut_np = pad_np[:, :keep, :]
            pad_chunks.append(pad_np); cut_chunks.append(cut_np)
            pad_pool.append(pad_np.mean(axis=1)); cut_pool.append(cut_np.mean(axis=1))
            pad_off.append(pad_off[-1] + MAX_FRAMES); cut_off.append(cut_off[-1] + keep)
            ids.append(rec["utt_id"]); keeps.append(keep); n_done += 1
        if not ids:
            continue
        np.savez(pad_dir / f"{spk}.npz", frames=np.concatenate(pad_chunks, axis=1),
                 offsets=np.array(pad_off, dtype=np.int32),
                 pooled=np.stack(pad_pool, axis=0), utt_ids=np.array(ids))
        np.savez(cut_dir / f"{spk}.npz", frames=np.concatenate(cut_chunks, axis=1),
                 offsets=np.array(cut_off, dtype=np.int32),
                 pooled=np.stack(cut_pool, axis=0), utt_ids=np.array(ids))
        print(f"  {spk}: {len(ids):3d} utts  [{n_done}/{len(sel)}, "
              f"{n_done/max(time.time()-t0,1e-6):.1f} utt/s]")

    if n_done < 0.9 * len(sel):
        raise SystemExit(f"only {n_done}/{len(sel)} utterances extracted — below the 90% gate")
    k = np.array(keeps)
    print(f"\nreal frames kept: mean {k.mean():.0f}, min {k.min()}, max {k.max()} of {MAX_FRAMES} "
          f"({100*(1-k.mean()/MAX_FRAMES):.1f}% is padding)")
    print(f"done: {n_done} utts in {time.time()-t0:.0f}s")
    print(f"  padded  -> {pad_dir}")
    print(f"  sliced  -> {cut_dir}")


if __name__ == "__main__":
    main()
