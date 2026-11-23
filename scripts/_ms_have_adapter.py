#!/usr/bin/env python3
"""Exit 0 if an adapter for (variant, seed, rank) already exists. Used to reuse the 2x2's seed-42 runs."""
import glob, json, os, sys

v, sd, rk = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for d in glob.glob(os.path.expanduser("~/dysarthria/models/layerprobe_lora/*/")):
    c = os.path.join(d, "_run_config.json")
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    if (cfg.get("variant") == v and cfg.get("seed") == sd
            and int(cfg.get("rank", -1)) == rk
            and os.path.isdir(os.path.join(d, "adapter"))):
        sys.exit(0)
sys.exit(1)
