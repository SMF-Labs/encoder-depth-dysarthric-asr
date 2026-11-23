#!/usr/bin/env python3
"""Print '<label>\t<run_dir>' for the four alpha-matched 2x2 cells at a given seed.

Every cell is required to satisfy alpha == 2*rank, and that is ASSERTED rather than assumed: the whole
point of this run is that the earlier 2x2 did not hold alpha/r fixed, so a cell that silently resolved
to an alpha=128 adapter would reproduce the very confound being removed.
"""
import glob, json, os, sys

seed = int(sys.argv[1])
WANT = [("upper", 64, 128), ("all", 21, 42), ("upper", 192, 384), ("all", 64, 128)]
found = {}
for d in sorted(glob.glob(os.path.expanduser("~/dysarthria/models/layerprobe_lora/*/"))):
    c = os.path.join(d, "_run_config.json")
    if not os.path.exists(c):
        continue
    try:
        cfg = json.load(open(c))
    except Exception:
        continue
    if cfg.get("seed") != seed or not os.path.isdir(os.path.join(d, "adapter")):
        continue
    key = (cfg.get("variant"), int(cfg.get("rank", -1)), int(cfg.get("alpha", -1)))
    if key in WANT and key not in found:
        found[key] = d.rstrip("/")
for v, rk, al in WANT:
    if (v, rk, al) in found:
        assert al == 2 * rk, "alpha/r not matched for %s r=%d alpha=%d" % (v, rk, al)
        print("%s_r%d\t%s" % (v, rk, found[(v, rk, al)]))
