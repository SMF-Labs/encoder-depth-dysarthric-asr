#!/usr/bin/env python3
"""Print '<band>\t<run_dir>' for the four Whisper x CDSD bands at a given seed.

Seed 42 is PINNED to the exact directories the published sweep used. `middle` has two seed-42 run dirs
on disk and resolving it by config would be a coin flip between them, which would silently change the
seed-42 column away from the numbers already reported in the paper.
"""
import glob, json, os, sys

seed = int(sys.argv[1])
PINNED_42 = {
    "lower": "lower__617a19ba55",
    "middle": "middle__2233834c1b",
    "upper": "upper__8899948325",
    "all": "all__649a6f2047",
}
root = os.path.expanduser("~/dysarthria/models/layerprobe_lora_whisper")
if seed == 42:
    for b in ("lower", "middle", "upper", "all"):
        d = os.path.join(root, PINNED_42[b])
        if os.path.isdir(os.path.join(d, "adapter")):
            print("%s\t%s" % (b, d))
    sys.exit(0)

found = {}
for d in sorted(glob.glob(root + "/*/")):
    c = os.path.join(d, "_run_config.json")
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    if (cfg.get("corpus") == "cdsd" and cfg.get("seed") == seed
            and int(cfg.get("rank", -1)) == 64
            and os.path.isdir(os.path.join(d, "adapter"))):
        b = cfg.get("variant")
        if b in ("lower", "middle", "upper", "all") and b not in found:
            found[b] = d.rstrip("/")
for b in ("lower", "middle", "upper", "all"):
    if b in found:
        print("%s\t%s" % (b, found[b]))
