#!/usr/bin/env python3
"""Exit 0 if an adapter for (variant, rank, alpha, seed) exists.

Alpha is part of the key on purpose. The original 2x2 trained `upper` r=192 and `all` r=21 at the
DEFAULT alpha=128, which is what this experiment exists to replace; matching on (variant, rank, seed)
alone would find those and silently skip the retrain.
"""
import glob, json, os, sys

v, rk, al, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
for d in glob.glob(os.path.expanduser("~/dysarthria/models/layerprobe_lora/*/")):
    c = os.path.join(d, "_run_config.json")
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    if (cfg.get("variant") == v and int(cfg.get("rank", -1)) == rk
            and int(cfg.get("alpha", -1)) == al and cfg.get("seed") == sd
            and os.path.isdir(os.path.join(d, "adapter"))):
        sys.exit(0)
sys.exit(1)
