#!/usr/bin/env bash
# Qwen3-ASR matched-volume probe: extract at the FULL 47,687-utt manifest, then sweep all 25 taps.
#
# Purpose: the Whisper volume run (2026-08-18) showed the post-minimum rise is strongly volume-
# dependent (+6.56 pp at 6,200 train -> +2.07 pp at 33,617). Qwen was only ever measured at 6,200
# (+1.85 pp). Until Qwen is measured at the SAME volume we cannot claim the late-stack shapes differ.
#
# Two design points, both learned the hard way on the Whisper run:
#   * NEW TAG (qwen_full). The existing `base/` shards hold only the 8,800-utt subset per speaker;
#     reusing that tag would hit the shard-exists skip and silently probe partial data.
#   * ONE PROCESS PER GROUP of 5 layers. On the GB10's unified memory a layer's allocation does not
#     fully release before the next load, which SIGKILLed the single-process Whisper sweep. Qwen is
#     ~2.5x lighter per layer than Whisper (25 taps x 1024 over 77 ms frames vs 13 x 768 over 20 ms),
#     so 5 layers fit comfortably in one process and cut the feature re-reads from 25 passes to 5.

# Working tree these runs read and write. Required: a wrong default would silently run
# against the wrong tree.
DYS_ROOT="${DYS_ROOT:?set DYS_ROOT to the dysarthria working tree, e.g. export DYS_ROOT=$HOME/dysarthria}"

set -uo pipefail
cd "$DYS_ROOT" || exit 1

TAG=qwen_full
MANIFEST=results/probe/cdsd_probe_manifest_FULL.jsonl
LOG="$DYS_ROOT"/logs/qwen_volume_perlayer_2026-08-19.log
DONE="$DYS_ROOT"/logs/qwen_volume_perlayer.DONE
rm -f "$DONE"

{
  echo "=== qwen matched-volume probe start $(date) ==="

  if [ ! -f "$MANIFEST" ]; then echo "FATAL: manifest missing: $MANIFEST"; echo "manifest" > "$DONE"; exit 1; fi
  echo "manifest rows: $(wc -l < "$MANIFEST")"

  # ML-05 disk gate: ~81 GB of features expected; refuse to start without real headroom.
  AVAIL=$(df -BG --output=avail "$DYS_ROOT" | tail -1 | tr -dc '0-9')
  echo "disk available: ${AVAIL} GB"
  if [ "${AVAIL:-0}" -lt 150 ]; then echo "FATAL: need >=150 GB free, have ${AVAIL} GB"; echo "disk" > "$DONE"; exit 1; fi

  source venv_qwen3asr/bin/activate

  echo "########## PHASE 1: extraction @ $(date) ##########"
  python -u scripts/layerprobe/extract_layer_features.py \
      --manifest "$MANIFEST" \
      --model Qwen/Qwen3-ASR-1.7B \
      --tag "$TAG"
  rc=$?
  echo "extraction rc=$rc"
  if [ $rc -ne 0 ]; then echo "FATAL: extraction failed"; echo "extract" > "$DONE"; exit 1; fi

  NSHARD=$(ls results/probe/features/${TAG}/*.npz 2>/dev/null | wc -l)
  echo "shards written: $NSHARD"
  if [ "$NSHARD" -lt 40 ]; then echo "FATAL: expected ~44 speaker shards, got $NSHARD"; echo "shards" > "$DONE"; exit 1; fi
  du -sh results/probe/features/${TAG}/

  echo "########## PHASE 2: 25-tap sweep @ $(date) ##########"
  fail=0
  for G in 0 1 2 3 4; do
    LO=$((G * 5)); HI=$((LO + 4))
    OUT=results/probe/qwen_volume_g${G}.csv
    if [ -f "$OUT" ]; then echo "group $G (layers $LO-$HI): already done, skipping"; continue; fi
    echo "########## group $G: layers $LO-$HI @ $(date) ##########"
    python -u scripts/layerprobe/train_probe_b_ctc.py \
        --tag "$TAG" \
        --manifest "$MANIFEST" \
        --layers $(seq -s' ' $LO $HI) \
        --layer-chunk 5 \
        --batch-size 64 \
        --out "$OUT"
    rc=$?
    echo "group $G rc=$rc"
    [ $rc -ne 0 ] && fail=$((fail+1))
    sleep 5
  done

  echo "groups failed: $fail"
  echo "$fail" > "$DONE"
  echo "=== QWEN MATCHED-VOLUME PROBE DONE $(date) ==="
} >> "$LOG" 2>&1
