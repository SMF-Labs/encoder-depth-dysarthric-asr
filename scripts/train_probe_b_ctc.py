#!/usr/bin/env python3
"""Probe B — layer-wise phoneme recognition (CTC) on frozen Qwen3-ASR encoder representations.

Replicates Muthu et al. (arXiv:2608.01865) §III.D.2 / §IV.B: for each encoder layer, a lightweight CTC
probe is trained on frozen hidden states and scored by phoneme error rate. Labels are pinyin
initial/final/tone units (tong2 -> t, ong, T2). PER is reported both tone-sensitive (over the full unit
sequence) and tone-agnostic (tone tokens stripped from BOTH hypothesis and reference before the edit
distance), so a single trained probe per layer yields both numbers — as in the paper, where tone markers
are "stripped before edit-distance computation".

Probe architecture, matching the paper: layer norm -> bottleneck projection (128) -> residual temporal
convolution -> linear classifier; dropout 0.3, blank bias 0.0. Optimised with CTC loss (Adam, lr 1e-3,
batch 256, 10 epochs, grad clip 1.0). Model selection on tone-sensitive PER over the val split.

Splits are speaker-disjoint (70/15/15 over speakers), so val and test speakers are unseen.

Usage (DGX, venv_qwen3asr):
    python scripts/layerprobe/train_probe_b_ctc.py --tag base
    python scripts/layerprobe/train_probe_b_ctc.py --tag zh_aug --layers 0 8 14 24
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(os.environ.get("DYSARTHRIA_BASE", Path(__file__).resolve().parents[2]))
PROBE_DIR = BASE / "results" / "probe"

BLANK = 0

# Set from --units-key / --no-tone-split. Defaults preserve the CDSD behaviour exactly.
UNITS_KEY = "units_tone"
TONE_SPLIT = True  # CTC blank occupies index 0; real units start at 1


class CTCProbe(nn.Module):
    def __init__(self, d_in: int, vocab: int, bottleneck: int = 128, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, bottleneck)
        self.conv = nn.Conv1d(bottleneck, bottleneck, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(bottleneck, vocab)
        nn.init.zeros_(self.out.bias)  # blank bias 0.0

    def forward(self, x):  # (B, T, D) -> (B, T, V)
        h = self.proj(self.norm(x))
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)  # residual temporal conv
        return self.out(self.drop(self.act(h)))


def levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_layers(tag: str, manifest_rows: list[dict], layers: list[int], feat_root: Path):
    """Load frame sequences for several layers at once, grouped by layer then split.

    Each .npz stores `frames` as one (25, sum_T, D) array, and numpy materialises the whole array
    before indexing — so reading one layer at a time would re-read the entire 15 GB feature set on
    every layer. Loading a chunk of layers per pass keeps the I/O proportional to the number of
    chunks rather than the number of layers.
    """
    by_spk: dict[str, list[dict]] = {}
    for r in manifest_rows:
        by_spk.setdefault(r["speaker_id"], []).append(r)

    out = {ly: {"train": [], "val": [], "test": []} for ly in layers}
    for spk, rows in by_spk.items():
        shard = feat_root / tag / f"{spk}.npz"
        if not shard.exists():
            continue
        with np.load(shard, allow_pickle=True) as z:
            frames = z["frames"]  # (25, sum_T, D) — read once per shard per chunk
            offsets = z["offsets"]
            ids = list(z["utt_ids"])
            index = {u: i for i, u in enumerate(ids)}
            for ly in layers:
                plane = frames[ly]
                for r in rows:
                    i = index.get(r["utt_id"])
                    if i is None:
                        continue
                    seq = plane[offsets[i]:offsets[i + 1]]
                    if seq.shape[0]:
                        # .copy() and NOT np.ascontiguousarray(): the slice is already C-contiguous,
                        # so ascontiguousarray returns the SAME OBJECT, a view whose .base is the
                        # shard's entire (n_taps, sum_T, D) array. Storing those views pins every
                        # shard in memory, so loading ONE layer retained the whole feature set —
                        # 87 GB instead of the 8.2 GB the layer actually needs, which is what drove
                        # the probe into swap on 2026-08-21. .copy() forces a real copy.
                        # speaker_id is carried so §5.5's per-layer speaker bootstrap is possible;
                        # every consumer unpacks three elements.
                        out[ly][r["split"]].append((seq.copy(), r[UNITS_KEY], spk))
    return out


def encode(units: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab[u] for u in units if u in vocab]


def batches(items, size, shuffle, rng=None):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


def collate(batch, device):
    lens = [b[0].shape[0] for b in batch]
    T = max(lens)
    D = batch[0][0].shape[1]
    x = torch.zeros(len(batch), T, D, dtype=torch.float32)
    for i, (seq, *_) in enumerate(batch):
        x[i, : seq.shape[0]] = torch.from_numpy(seq.astype(np.float32))
    return x.to(device), torch.tensor(lens, dtype=torch.long)


def greedy_decode(logits: torch.Tensor, length: int) -> list[int]:
    ids = logits[:length].argmax(dim=-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


def evaluate(model, items, vocab, inv_vocab, device, batch_size=64, per_utt=None):
    """Aggregate PER. If `per_utt` is a list, per-utterance error/length counts are appended to it.

    The aggregate return value is unchanged whether or not `per_utt` is supplied — the extra work is
    purely recording, so enabling it cannot move a headline number.
    """
    model.eval()
    err_t = tot_t = err_n = tot_n = 0
    with torch.no_grad():
        for batch in batches(items, batch_size, False):
            x, lens = collate(batch, device)
            logits = model(x).cpu()
            for i, (_, units, spk) in enumerate(batch):
                hyp = [inv_vocab[j] for j in greedy_decode(logits[i], int(lens[i]))]
                ref = [u for u in units if u in vocab]
                e_t = levenshtein(hyp, ref)
                err_t += e_t
                tot_t += len(ref)
                # Mandarin only. On an ARPAbet inventory startswith("T") would delete T and TH,
                # so for non-tonal corpora the "tone-agnostic" column is simply the same rate.
                if TONE_SPLIT:
                    hyp_n = [u for u in hyp if not u.startswith("T")]
                    ref_n = [u for u in ref if not u.startswith("T")]
                else:
                    hyp_n, ref_n = hyp, ref
                e_n = levenshtein(hyp_n, ref_n)
                err_n += e_n
                tot_n += len(ref_n)
                if per_utt is not None:
                    per_utt.append({"speaker_id": spk, "err_tone": e_t, "ref_tone": len(ref),
                                    "err_notone": e_n, "ref_notone": len(ref_n)})
    return 100.0 * err_t / max(tot_t, 1), 100.0 * err_n / max(tot_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="feature tag: base or zh_aug")
    ap.add_argument("--manifest", type=Path, default=PROBE_DIR / "cdsd_probe_manifest.jsonl")
    ap.add_argument("--feat-root", type=Path, default=PROBE_DIR / "features")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: all 25 taps")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer-chunk", type=int, default=5,
                    help="layers held in memory per pass over the feature shards")
    ap.add_argument("--out", type=Path, default=PROBE_DIR / "qwen3asr_zh_probe_layerwise.csv")
    ap.add_argument("--units-key", default="units_tone",
                    help="manifest field holding the target phone sequence: units_tone for CDSD "
                         "(pinyin), units for UASpeech (ARPAbet)")
    ap.add_argument("--no-tone-split", action="store_true",
                    help="disable the Mandarin tone-stripping second metric; required for ARPAbet, "
                         "where startswith('T') would delete T and TH")
    ap.add_argument("--per-utt-dir", default=None,
                    help="if set, write per-utterance test error counts here for speaker bootstrap")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reserve a CUDA slab BEFORE features are loaded. The GB10 has UNIFIED memory: the feature
    # arrays live in the same pool CUDA allocates from, so loading ~68 GB of activations leaves
    # CUDA unable to allocate even this few-MB probe, and the run dies at model construction with
    # "CUDA error: out of memory" while `free` still reports tens of GB available. Allocating here
    # and immediately releasing hands the block to torch's caching allocator, which keeps it
    # reserved from the OS, so every later per-layer model/batch allocation is served from cache.
    global UNITS_KEY, TONE_SPLIT
    UNITS_KEY = args.units_key
    TONE_SPLIT = not args.no_tone_split
    print(f"  units key: {UNITS_KEY}  tone split: {TONE_SPLIT}")

    if device == "cuda":
        _gb = float(os.environ.get("PROBE_CUDA_RESERVE_GB", "6"))
        _slab = torch.empty(int(_gb * 1e9 // 2), dtype=torch.float16, device="cuda")
        del _slab
        print(f"  reserved {_gb:.0f} GB CUDA slab before feature load")

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]

    counts = Counter(u for r in rows if r["split"] == "train" for u in r[UNITS_KEY])
    units = sorted(counts)
    vocab = {u: i + 1 for i, u in enumerate(units)}
    inv_vocab = {i + 1: u for i, u in enumerate(units)}
    print(f"vocab: {len(vocab)} units (+blank)  |  device={device}")

    layers = args.layers if args.layers is not None else list(range(25))
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    chunks = [layers[i:i + args.layer_chunk] for i in range(0, len(layers), args.layer_chunk)]
    for chunk in chunks:
        print(f"\nloading layers {chunk} ...")
        t_load = time.time()
        loaded = load_layers(args.tag, rows, chunk, args.feat_root)
        print(f"  loaded in {time.time()-t_load:.0f}s")

        for layer in chunk:
            t0 = time.time()
            data = loaded[layer]
            tr = [(s, u, k) for s, u, k in data["train"] if encode(u, vocab)]
            va = [(s, u, k) for s, u, k in data["val"] if encode(u, vocab)]
            te = [(s, u, k) for s, u, k in data["test"] if encode(u, vocab)]
            if not tr or not te:
                print(f"layer {layer}: no data, skipping")
                continue
            _run_layer(args, layer, tr, va, te, vocab, inv_vocab, device, results, t0)
        loaded.clear()

    print(f"\nwrote {len(results)} rows -> {args.out}")


def _run_layer(args, layer, tr, va, te, vocab, inv_vocab, device, results, t0):
    d_in = tr[0][0].shape[1]
    torch.manual_seed(args.seed)
    model = CTCProbe(d_in, len(vocab) + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    rng = np.random.default_rng(args.seed)

    best_per, best_state = 1e9, None
    for _ in range(args.epochs):
        model.train()
        for batch in batches(tr, args.batch_size, True, rng):
            x, lens = collate(batch, device)
            targets = [encode(u, vocab) for _, u, _ in batch]
            tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
            flat = torch.tensor([i for t in targets for i in t], dtype=torch.long)
            logp = model(x).log_softmax(-1).transpose(0, 1)  # (T, B, V)
            loss = ctc(logp, flat, lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        per_t, _ = evaluate(model, va, vocab, inv_vocab, device)
        if per_t < best_per:
            best_per = per_t
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    per_utt = [] if args.per_utt_dir else None
    per_tone, per_notone = evaluate(model, te, vocab, inv_vocab, device, per_utt=per_utt)
    if per_utt is not None:
        pu_dir = Path(args.per_utt_dir)
        pu_dir.mkdir(parents=True, exist_ok=True)
        pu_path = pu_dir / f"{args.tag}_L{layer:02d}_seed{args.seed}.csv"
        with open(pu_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["speaker_id", "err_tone", "ref_tone",
                                               "err_notone", "ref_notone"])
            w.writeheader()
            w.writerows(per_utt)
        print(f"  per-utt -> {pu_path.name} ({len(per_utt)} rows)")
    results.append(
        {"model": args.tag, "seed": args.seed, "layer": layer,
         "per_tone_sensitive": round(per_tone, 3),
         "per_tone_agnostic": round(per_notone, 3), "val_per": round(best_per, 3),
         "n_train": len(tr), "n_test": len(te)}
    )
    print(f"layer {layer:2d}: test PER tone-sensitive={per_tone:6.2f}  "
          f"tone-agnostic={per_notone:6.2f}  (val {best_per:6.2f})  [{time.time()-t0:.0f}s]")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)


if __name__ == "__main__":
    main()
