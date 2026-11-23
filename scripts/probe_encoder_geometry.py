#!/usr/bin/env python3
"""Establish Qwen3-ASR encoder geometry for the layer-probe study.

Reports, for a correctly-loaded checkpoint:
  - encoder layer count and the number of hidden_states returned
  - encoder output frame rate (frames/s and ms/frame), needed to map MFA phone
    intervals onto encoder frames
  - whether the frame rate is stable across utterance durations
  - LoRA-targetable linear modules per encoder layer

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/probe_encoder_geometry.py Qwen/Qwen3-ASR-1.7B <wav> [<wav> ...]
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen3asr_compat import cfg_get, encoder_of, load_qwen3_asr, make_encoder_inputs  # noqa: E402

SR = 16000


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-ASR-1.7B"
    wavs = sys.argv[2:]

    print(f"loading {model_id} ...")
    model, cfg, report = load_qwen3_asr(model_id, dtype=torch.float32)
    print(f"  load OK — missing={len(report['missing'])} unexpected={len(report['unexpected'])}")
    if report["unexpected"]:
        print(f"  unexpected (ignored): {report['unexpected'][:5]}")

    enc = encoder_of(model)
    layers = enc.layers
    print(f"\nencoder: {type(enc).__name__}  layers={len(layers)}  "
          f"d_model={cfg_get(cfg, 'audio_config', 'd_model')} "
          f"heads={cfg_get(cfg, 'audio_config', 'encoder_attention_heads')} "
          f"ffn={cfg_get(cfg, 'audio_config', 'encoder_ffn_dim')} "
          f"output_dim={cfg_get(cfg, 'audio_config', 'output_dim')}")
    print(f"  n_window={getattr(enc, 'n_window', '?')} n_window_infer={getattr(enc, 'n_window_infer', '?')}")

    print("\nLoRA-targetable linears per encoder layer:")
    for n, m in layers[0].named_modules():
        if isinstance(m, torch.nn.Linear):
            print(f"    {n:24s} {tuple(m.weight.shape)}")

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)

    print("\nframe-rate probe:")
    rows = []
    for src in (wavs or [None, None, None]):
        if src:
            audio, sr = sf.read(src, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                import torchaudio
                audio = torchaudio.transforms.Resample(sr, SR)(
                    torch.from_numpy(audio).unsqueeze(0)).squeeze(0).numpy()
            label = Path(src).name
        else:
            dur = {0: 2.0, 1: 5.0, 2: 11.0}[len(rows)]
            audio = np.zeros(int(SR * dur), dtype="float32")
            label = f"silence-{dur:g}s"

        dur = len(audio) / SR
        feats, mask = make_encoder_inputs(proc, audio, SR)
        with torch.no_grad():
            out = enc(feats, mask, output_hidden_states=True)
        hs = out.hidden_states
        T_pad = hs[-1].shape[-2]
        # valid frames = mel frames that were real, carried through the conv/window stack
        n_mel_valid = int(mask.sum().item())
        ratio = T_pad / (feats.shape[-1])
        T_valid = int(round(n_mel_valid * ratio))
        rows.append((label, dur, feats.shape, T_pad, T_valid, len(hs)))
        print(f"  {label:34s} dur={dur:6.2f}s mel={tuple(feats.shape)} "
              f"padded_T={T_pad} valid_T~{T_valid} n_hidden={len(hs)} "
              f"-> {T_valid/dur:7.2f} fps ({1000*dur/max(T_valid,1):5.2f} ms/frame)")

    print(f"\nhidden_states count = {rows[0][5]} (expect layers+1 = {len(layers)+1}); "
          f"index 0 = encoder input, 1..{len(layers)} = block outputs")
    print("mel hop is 10 ms (WhisperFeatureExtractor hop_length=160 @ 16 kHz); "
          "divide mel frames by the observed ratio to get encoder frames.")


if __name__ == "__main__":
    main()
