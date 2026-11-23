#!/usr/bin/env bash
# One layer per PROCESS. The loader's per-layer allocation (~66 GB on unified memory) does not fully
# release before the next layer's load begins, so a single 13-layer process gets OOM-killed at the
# first transition. Separate invocations return memory to the OS between layers.

# Working tree these runs read and write. Required: a wrong default would silently run
# against the wrong tree.
DYS_ROOT="${DYS_ROOT:?set DYS_ROOT to the dysarthria working tree, e.g. export DYS_ROOT=$HOME/dysarthria}"

set -uo pipefail
cd "$DYS_ROOT" || exit 1
LOG="$DYS_ROOT"/logs/whisper_volume_perlayer_2026-08-18.log
DONE="$DYS_ROOT"/logs/whisper_volume_perlayer.DONE
rm -f "$DONE"
{
  echo "=== per-layer volume probe start $(date) ==="
  source venv_qwen3asr/bin/activate
  fail=0
  for L in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    OUT=results/probe/volume_layer_${L}.csv
    if [ -f "$OUT" ]; then echo "layer $L: already done, skipping"; continue; fi
    echo "########## layer $L @ $(date) ##########"
    python -u scripts/layerprobe/train_probe_b_ctc.py --tag whisper_full       --manifest results/probe/cdsd_probe_manifest_FULL.jsonl       --layers $L --layer-chunk 1 --batch-size 64 --out "$OUT"
    rc=$?
    echo "layer $L rc=$rc"
    [ $rc -ne 0 ] && fail=$((fail+1))
    sleep 5
  done
  echo "layers failed: $fail"
  echo "$fail" > "$DONE"
  echo "=== PER-LAYER VOLUME PROBE DONE $(date) ==="
} >> "$LOG" 2>&1
