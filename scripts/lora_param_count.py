#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trainable-parameter counts for every LoRA configuration in Paper 10a.

WHY THIS SHIPS. Section 6.7's control turns on two configurations having the SAME trainable-parameter
count at different placements. That is an arithmetic claim about the adapter geometry, not a measurement,
so it should be checkable without a GPU, a checkpoint, or the corpus. Run this and compare with Tables 19
and 20.

THE ARITHMETIC. A LoRA adapter on a d_in x d_out projection adds r*(d_in + d_out) trainable parameters
(the A and B factors; bias is 'none' throughout, and dropout has no parameters). Every adapter here
targets the four attention projections q_proj, k_proj, v_proj and out_proj, all square at the encoder
width, so each adapted layer costs 4 * r * 2 * d_model. Only encoder blocks are adapted; the decoder,
the multi-modal projector and the embeddings are frozen.

    total = 4 * n_layers * r * 2 * d_model

Widths and depths are properties of the released base checkpoints: Qwen3-ASR-1.7B's audio tower is 24
blocks at width 1024, Whisper-small's encoder is 12 blocks at width 768.
"""
import argparse

MODELS = {
    "qwen3asr-1.7b":  dict(d_model=1024, n_blocks=24, tower="audio_tower"),
    "whisper-small":  dict(d_model=768,  n_blocks=12, tower="encoder"),
}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


def bands(n_blocks):
    """The four placements. Thirds of the stack, plus all-block."""
    t = n_blocks // 3
    return {
        "lower":  list(range(0, t)),
        "middle": list(range(t, 2 * t)),
        "upper":  list(range(2 * t, n_blocks)),
        "all":    list(range(0, n_blocks)),
    }


def count(d_model, layers, r, n_proj=len(PROJECTIONS)):
    # r * (d_in + d_out) per projection; square projections give d_in = d_out = d_model.
    return n_proj * len(layers) * r * 2 * d_model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen3asr-1.7b")
    ap.add_argument("--ranks", type=int, nargs="+", default=[21, 64, 192])
    args = ap.parse_args()

    m = MODELS[args.model]
    bs = bands(m["n_blocks"])
    print(f"{args.model}: {m['n_blocks']} {m['tower']} blocks at width {m['d_model']}, "
          f"targeting {', '.join(PROJECTIONS)}\n")
    print(f"    {'band':8s} {'blocks':>7s}  " + "  ".join(f"r={r:<11d}" for r in args.ranks))
    for b, layers in bs.items():
        cells = "  ".join(f"{count(m['d_model'], layers, r):>13,d}" for r in args.ranks)
        print(f"    {b:8s} {len(layers):>7d}  {cells}")

    if args.model == "qwen3asr-1.7b":
        print("\n  The two budget-matched pairs in Section 6.7:")
        for (b1, r1), (b2, r2) in ((("upper", 192), ("all", 64)), (("upper", 64), ("all", 21))):
            n1 = count(m["d_model"], bs[b1], r1)
            n2 = count(m["d_model"], bs[b2], r2)
            verdict = "EXACT match" if n1 == n2 else f"within {abs(n1-n2)/max(n1,n2):.1%}"
            print(f"    {b1} r={r1:<3d} {n1:>12,d}   vs   {b2} r={r2:<3d} {n2:>12,d}   {verdict}")
        print("\n  Alpha is 2r in the alpha-matched replication, so the update scaling alpha/r is 2 in\n"
              "  every cell and placement is the only difference. The originally published 2x2 left\n"
              "  alpha at 128 for every rank; see Section 6.7.")


if __name__ == "__main__":
    main()
