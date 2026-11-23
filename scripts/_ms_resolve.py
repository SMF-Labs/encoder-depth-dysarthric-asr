#!/usr/bin/env python3
"""Print '<band>\t<run_dir>' for each of the four bands at (seed, rank).

Resolves by reading each run's stored config rather than by guessing the signature path: two runs of the
same band can differ only in a hash, and scoring the wrong one would be attributed to the seed.
"""
import glob, json, os, sys

seed, rk = int(sys.argv[1]), int(sys.argv[2])
found = {}
for d in sorted(glob.glob(os.path.expanduser("~/dysarthria/models/layerprobe_lora/*/"))):
    c = os.path.join(d, "_run_config.json")
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    if (cfg.get("seed") == seed and int(cfg.get("rank", -1)) == rk
            and os.path.isdir(os.path.join(d, "adapter"))):
        b = cfg.get("variant")
        if b in ("lower", "middle", "upper", "all") and b not in found:
            found[b] = d.rstrip("/")
for b in ("lower", "middle", "upper", "all"):
    if b in found:
        print("%s\t%s" % (b, found[b]))
