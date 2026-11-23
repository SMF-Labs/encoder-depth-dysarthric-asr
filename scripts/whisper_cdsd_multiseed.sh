#!/usr/bin/env bash
# Whisper-small x CDSD at three matched seeds (reviewer's mandatory item 1).
#
# WHY. The CDSD cross-model reversal rests on two cells. Qwen3-ASR x CDSD now carries three seeds; the
# Whisper-small x CDSD half is still a single run, so the comparison has asymmetric seed evidence - and
# this paper's own lesson is that ~1 pp band differences can vanish once seeds are varied. Adding seeds
# 43 and 44 completes the seed-balanced design behind the headline reversal.
#
# Seed 42 already exists for all four bands and is REUSED, pinned to the exact run directories the
# published sweep used (`middle` has two seed-42 dirs; resolving by config would pick between them at
# random and silently move the seed-42 column).
#
# Scoring uses --decode fixed, the repetition-controlled decode of the 2026-08-25 correction, so these
# numbers are commensurable with the corrected Whisper results already in the paper.
set -uo pipefail
# Repository root of the working tree these runs read and write. Override to relocate.
DYS_ROOT="${DYS_ROOT:?set DYS_ROOT to the dysarthria working tree, e.g. export DYS_ROOT=$HOME/dysarthria}"
cd "$DYS_ROOT" || exit 1

MAN=results/probe/cdsd_lora_manifest.jsonl
LOG=logs/whisper_cdsd_multiseed.log
DONE=logs/whisper_cdsd_multiseed.DONE
MARK=logs/whisper_cdsd_multiseed_markers
mkdir -p "$MARK"; rm -f "$DONE"

CELLS=("upper:43" "middle:43" "upper:44" "middle:44" "all:43" "all:44" "lower:43" "lower:44")

{
  echo "=== whisper x cdsd multiseed start $(date) ==="
  AVAIL=$(df -BG --output=avail "$DYS_ROOT" | tail -1 | tr -dc '0-9')
  echo "disk available: ${AVAIL} GB"
  if [ "${AVAIL:-0}" -lt 60 ]; then echo "FATAL: need >=60 GB"; echo disk > "$DONE"; exit 1; fi
  source venv_qwen3asr/bin/activate || { echo "FATAL venv"; echo venv > "$DONE"; exit 1; }

  fail=0
  for cell in "${CELLS[@]}"; do
    v="${cell%%:*}"; sd="${cell##*:}"
    tag="w_${v}_s${sd}"
    if [ -f "$MARK/$tag.done" ]; then echo "SKIP  $tag (marker)"; continue; fi
    echo "########## TRAIN $tag @ $(date +%H:%M:%S) ##########"
    python -u scripts/layerprobe/train_lora_whisper_sweep.py \
        --variant "$v" --corpus cdsd --manifest "$MAN" --seed "$sd"
    rc=$?
    echo "$tag rc=$rc"
    if [ $rc -eq 0 ]; then date > "$MARK/$tag.done"; else fail=$((fail+1)); fi
  done
  echo "PHASE 1 complete, cells failed: $fail"

  for sd in 42 43 44; do
    OUT="results/probe/whisper_cdsd_multiseed_s${sd}.csv"
    if [ -f "$OUT" ]; then echo "SKIP eval s$sd (exists)"; continue; fi
    mapfile -t DIRS < <(python3 scripts/layerprobe/_wm_resolve.py "$sd")
    if [ "${#DIRS[@]}" -ne 4 ]; then
      echo "SKIP eval s$sd: resolved ${#DIRS[@]}/4 adapters"; continue
    fi
    ARGS=(--variant "base=NONE")
    for line in "${DIRS[@]}"; do
      b=$(printf '%s' "$line" | cut -f1)
      dir=$(printf '%s' "$line" | cut -f2)
      echo "  resolved $b -> $dir"
      ARGS+=(--variant "${b}_s${sd}=${dir}/adapter")
    done
    echo "########## EVAL seed $sd @ $(date +%H:%M:%S) ##########"
    python -u scripts/layerprobe/eval_whisper_variants.py \
        --decode fixed --manifest "$MAN" --split test --batch-size 16 \
        --out "$OUT" --preds-dir "results/probe/whisper_cdsd_multiseed_preds/s${sd}" "${ARGS[@]}"
    echo "eval s$sd rc=$?"
  done

  echo "$fail" > "$DONE"
  echo "=== WHISPER CDSD MULTISEED DONE $(date) ==="
} >> "$LOG" 2>&1
