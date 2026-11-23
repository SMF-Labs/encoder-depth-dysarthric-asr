#!/usr/bin/env python3
"""Generate the four figures for Paper 10a from the released aggregate CSVs.

Numbered in order of appearance in the manuscript:
  Fig 1 (S5.2)  divergence + difficulty probes
  Fig 2 (S5.5)  both depth profiles vs relative depth - the minima coincide
  Fig 3 (S5.5)  post-minimum rise at two probe-training volumes - the data artefact
  Fig 4 (S6.6)  both band sweeps side by side - placement differs by architecture

Figures 2 and 4 are the argument: same knee, different winner. Figure 3 defends Figure 2
against "your curves have no error bars" by showing replication across probe volumes instead.

Reads only files in ../results/. No GPU, no corpus access.

Usage:
    python scripts/make_figures.py [--outdir ../figures] [--format pdf png]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# SMF brand palette
NAVY, ORANGE, LTBLUE, GREY = "#0B1F5C", "#F04E23", "#4DB8C8", "#8A8F9A"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def read(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def curve(name, layer_key="layer", val_key="per_tone_sensitive"):
    rows = read(name)
    d = {int(r[layer_key]): float(r[val_key]) for r in rows}
    return sorted(d), [d[k] for k in sorted(d)]


def qwen_6200():
    d = {}
    for r in read("qwen3asr_zh_probe_layerwise.csv"):
        if r.get("model") == "base" and r.get("metric") == "PER_tone_sensitive":
            d[int(r["layer"])] = float(r["value"])
    return sorted(d), [d[k] for k in sorted(d)]


def annotate_min(ax, xs, ys, n_blocks, colour, dx, dy, ha="right"):
    i = min(range(len(ys)), key=lambda j: ys[j])
    rel = 100.0 * xs[i] / n_blocks
    ax.plot([rel], [ys[i]], marker="o", ms=7, mfc="white", mec=colour, mew=2, zorder=5)
    ax.annotate("L{}/{}  ({:.0f}% depth)".format(xs[i], n_blocks, rel),
                xy=(rel, ys[i]), xytext=(rel + dx, ys[i] + dy),
                ha=ha, fontsize=7.5, color=colour, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=colour, lw=0.8, alpha=0.6))
    return rel


def fig2_depth_profiles(outdir, fmts):
    qx, qy = curve("qwen_volume_probe_layerwise.csv")
    wx, wy = curve("whisper_volume_probe_layerwise.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([100 * x / 24 for x in qx], qy, "-o", color=NAVY, ms=3.2, lw=1.6,
            label="Qwen3-ASR (24 blocks)")
    ax.plot([100 * x / 12 for x in wx], wy, "-s", color=ORANGE, ms=3.6, lw=1.6,
            label="Whisper-small (12 blocks)")
    # labels pushed to opposite sides so the two minima, which sit close together, stay legible
    rw = annotate_min(ax, wx, wy, 12, ORANGE, dx=-13, dy=13, ha="right")
    rq = annotate_min(ax, qx, qy, 24, NAVY, dx=3, dy=4.5, ha="left")
    lo, hi = min(rq, rw) - 2.0, max(rq, rw) + 2.0
    ax.axvspan(lo, hi, color=LTBLUE, alpha=0.16, zorder=0)
    ax.set_xlabel("Relative encoder depth (%)")
    ax.set_ylabel("Phoneme error rate (%), tone-sensitive")
    ax.set_title("Both probes reach their minima in the late encoder", pad=8)
    ax.legend(loc="lower left", frameon=False)
    ax.set_xlim(-3, 108)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 3)
    ax.text((lo + hi) / 2, ymax + 1.2, "both minima late", ha="center", va="bottom",
            fontsize=7.5, color=LTBLUE, fontweight="bold")
    for f in fmts:
        fig.savefig(outdir / "fig2_depth_profiles.{}".format(f))
    plt.close(fig)
    print("  fig2: Qwen min {:.0f}% depth, Whisper min {:.0f}% depth".format(rq, rw))


def _multiseed(pattern):
    """Mean over per-seed CSVs, returned in the same shape as a single-run CSV row dict.

    Figure 4 previously read the single-seed Qwen sweep and the UNCORRECTED-decode Whisper sweep, so it
    kept showing a 103.6 Whisper baseline and single-run Qwen bars long after the tables had moved to
    three seeds. It now reads the same canonical per-seed CSVs the tables are built from.
    """
    import glob, statistics as st
    acc, spk = {}, {}
    for f in sorted(glob.glob(str(RESULTS / pattern))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            v = r["variant"]
            for sd in ("_s42", "_s43", "_s44"):
                v = v.replace(sd, "")
            try:
                acc.setdefault(v, []).append(float(r["cer"]))
            except (TypeError, ValueError):
                continue
            try:
                for k, val in json.loads(r["per_speaker_cer"]).items():
                    spk.setdefault(v, {}).setdefault(k, []).append(float(val))
            except Exception:
                pass
    out = {}
    for v, vals in acc.items():
        ps = {k: st.mean(x) for k, x in spk.get(v, {}).items()}
        out[v] = {"variant": v, "cer": st.mean(vals),
                  "per_speaker_cer": json.dumps(ps)}
    return out


def fig4_band_placement(outdir, fmts):
    q = _multiseed("cdsd_multiseed_s*.csv")
    w = _multiseed("whisper_cdsd_multiseed_s*.csv")
    bands = ["lower", "middle", "upper", "all"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, data, name in ((axes[0], q, "Qwen3-ASR"), (axes[1], w, "Whisper-small")):
        base = float(data["base"]["cer"])
        vals = [100.0 * float(data[b]["cer"]) / base for b in bands]
        best = min(range(len(vals)), key=lambda i: vals[i])
        cols = [ORANGE if i == best else NAVY for i in range(len(bands))]
        ax.bar(bands, vals, color=cols, width=0.62, zorder=3)
        ax.axhline(100, color=GREY, ls="--", lw=1.2, zorder=4)
        try:
            pb = json.loads(data["base"]["per_speaker_cer"])
        except Exception:
            pb = {}
        for i, b in enumerate(bands):
            try:
                ps = json.loads(data[b]["per_speaker_cer"])
            except Exception:
                continue
            pts = [100.0 * ps[k] / pb[k] for k in ps if k in pb and pb[k]]
            ax.scatter([i] * len(pts), pts, s=10, color="white", edgecolor=NAVY,
                       linewidth=0.6, alpha=0.9, zorder=6)
        for i, b in enumerate(bands):
            ax.text(i, vals[i] + 3.0, "{:.1f}".format(float(data[b]["cer"])),
                    ha="center", fontsize=7.5, fontweight="bold",
                    color=ORANGE if i == best else NAVY, zorder=7)
        ax.set_title("{}\nunadapted CER {:.1f}%".format(name, base), fontsize=9, pad=6)
        ax.set_xlabel("Adapted band")
    axes[0].set_ylabel("CER relative to unadapted (%)")
    fig.suptitle("Optimal placement differs by architecture", fontsize=10, y=1.03)
    legend = [Line2D([], [], marker="o", ls="", mfc="white", mec=NAVY, ms=5,
                     label="individual speaker"),
              Line2D([], [], color=ORANGE, lw=6, label="best band"),
              Line2D([], [], color=GREY, ls="--", lw=1.2, label="no adaptation")]
    axes[1].legend(handles=legend, loc="upper right", frameon=False, fontsize=7)
    for f in fmts:
        fig.savefig(outdir / "fig4_band_placement.{}".format(f))
    plt.close(fig)
    print("  fig4: per-band CER normalised to each architecture's own baseline")


def fig3_volume_artefact(outdir, fmts):
    q6x, q6y = qwen_6200()
    qfx, qfy = curve("qwen_volume_probe_layerwise.csv")
    w6x, w6y = curve("whisper_probe_layerwise.csv")
    wfx, wfy = curve("whisper_volume_probe_layerwise.csv")

    panels = [("6,200 probe-training utterances", [(q6x, q6y, 24), (w6x, w6y, 12)]),
              ("33,617 probe-training utterances", [(qfx, qfy, 24), (wfx, wfy, 12)])]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, (title, sets) in zip(axes, panels):
        # Both labels park in the right margin. The curves converge at matched volume, so any
        # in-plot placement collides with one line or the other.
        offsets = {"Qwen3-ASR": (105, 0), "Whisper-small": (105, 0)}
        for (xs, ys, nb), col, mk, lab in zip(sets, (NAVY, ORANGE), ("o", "s"),
                                              ("Qwen3-ASR", "Whisper-small")):
            ax.plot([100 * x / nb for x in xs], ys, "-", marker=mk, color=col,
                    ms=3.2, lw=1.5, label=lab)
            i = min(range(len(ys)), key=lambda j: ys[j])
            ax.plot([100 * xs[i] / nb], [ys[i]], marker=mk, ms=7, mfc="white",
                    mec=col, mew=1.8, zorder=5)
            rise = ys[-1] - ys[i]
            ox, oy = offsets[lab]
            ax.annotate("+{:.2f} pp".format(rise), xy=(100, ys[-1]),
                        xytext=(ox, ys[-1] + oy), fontsize=7.5, color=col,
                        fontweight="bold", ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7, alpha=0.55))
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel("Relative encoder depth (%)")
        ax.set_xlim(-3, 140)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel("Phoneme error rate (%)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Post-minimum rise: an apparent architectural difference that is a probe-data artefact",
                 fontsize=9.5, y=1.03)
    for f in fmts:
        fig.savefig(outdir / "fig3_volume_artefact.{}".format(f))
    plt.close(fig)
    print("  fig3: post-minimum rise at both volumes")


def fig1_divergence_difficulty(outdir, fmts):
    rows = read("qwen3asr_zh_condition_divergence.csv")
    strict = {int(r["layer"]): float(r["distance"]) for r in rows
              if r["analysis"] == "strict_paired" and r["pair"] == "d-r"}
    dist = {int(r["layer"]): float(r["distance"]) for r in rows
            if r["analysis"] == "distributional_global" and r["pair"] == "d-r"}
    pc = read("probeC_base_layerwise.csv")
    pl = [int(r["layer"]) for r in pc]
    pf = [float(r["macro_f1"]) for r in pc]
    ps = [float(r["std"]) for r in pc]

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.4), sharex=True)
    ax = axes[0]
    xs = sorted(strict)
    ax.plot(xs, [strict[k] for k in xs], "-o", color=NAVY, ms=3.2, lw=1.6,
            label="strict-paired (primary)")
    if dist:
        xs2 = sorted(dist)
        ax.plot(xs2, [dist[k] for k in xs2], "--", color=LTBLUE, lw=1.4,
                label="distributional")
    pk = max(strict, key=lambda k: strict[k])
    ax.plot([pk], [strict[pk]], marker="o", ms=7, mfc="white", mec=ORANGE, mew=2, zorder=5)
    ax.annotate("peak L{}".format(pk), xy=(pk, strict[pk]),
                xytext=(pk - 1.2, strict[pk] + 0.008), fontsize=7.5, color=ORANGE,
                fontweight="bold", ha="right")
    ax.set_ylabel("Cross-condition distance\n(1 - cosine)")
    ax.set_title("Divergence peaks mid-encoder, then collapses", fontsize=9, pad=6)
    ax.legend(loc="lower left", frameon=False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.012)

    ax = axes[1]
    ax.errorbar(pl, pf, yerr=ps, fmt="-o", color=ORANGE, ms=3.2, lw=1.6,
                ecolor=GREY, elinewidth=0.8, capsize=2)
    ax.set_xlabel("Encoder layer (Qwen3-ASR, 0 = encoder input)")
    ax.set_ylabel("Difficulty probe macro-F1")
    ax.set_title("Recognition difficulty is deepest-encoded", fontsize=9, pad=6)
    for f in fmts:
        fig.savefig(outdir / "fig1_divergence_difficulty.{}".format(f))
    plt.close(fig)
    print("  fig1: divergence peak L{}, Probe C max {:.3f}".format(pk, max(pf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=HERE.parent / "figures")
    ap.add_argument("--format", nargs="+", default=["pdf", "png"])
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print("writing to {}".format(args.outdir))
    fig1_divergence_difficulty(args.outdir, args.format)
    fig2_depth_profiles(args.outdir, args.format)
    fig3_volume_artefact(args.outdir, args.format)
    fig4_band_placement(args.outdir, args.format)
    print("done")


if __name__ == "__main__":
    main()
