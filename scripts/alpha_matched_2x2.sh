#!/usr/bin/env bash
# Alpha-matched placement x capacity 2x2 on CDSD (Qwen3-ASR), three seeds.
#
# WHY. The 2x2 in the paper matched TOTAL trainable parameters but left LoRA alpha at the default 128
# for every rank, so alpha/r was not matched: 0.667 for `upper` r=192 against 2.000 for `all` r=64 in
# the high-capacity pair, and 2.000 against 6.095 in the low-capacity one. The arms therefore differed
# in three things at once - where the capacity sits, how concentrated it is per block, and how strongly
# each update is scaled. This rebuilds the 2x2 with alpha = 2r everywhere, which removes the scaling
# term and leaves placement and per-block rank as the only differences.
#
# ONLY TWO CELLS NEED TRAINING. `upper` r=64 and `all` r=64 already run at alpha=128 = 2r, and the
# three-seed sweep produced both at seeds 42/43/44, so they are reused. What is missing is
# `upper` r=192 at alpha=384 and `all` r=21 at alpha=42 - 6 runs rather than 12.
#
# The alpha=128 versions of those two cells ARE STILL ON DISK from the original 2x2. Every lookup here
# keys on (variant, rank, alpha, seed) and the resolver asserts alpha == 2*rank, because resolving on
# (variant, rank, seed) alone would silently pick up the old adapters and reproduce the confound.
#
# Seed 42 is done first for both new cells so a complete alpha-matched 2x2 exists at one seed early.
set -uo pipefail
# Repository root of the working tree these runs read and write. Override to relocate.
DYS_ROOT="${DYS_ROOT:?set DYS_ROOT to the dysarthria working tree, e.g. export DYS_ROOT=$HOME/dysarthria}"
cd "$DYS_ROOT" || exit 1

MAN=results/probe/cdsd_lora_manifest.jsonl
LOG=logs/alpha_matched.log
DONE=logs/alpha_matched.DONE
MARK=logs/alpha_matched_markers
mkdir -p "$MARK"
rm -f "$DONE"

# variant:rank:alpha:seed
CELLS=("upper:192:384:42" "all:21:42:42" "upper:192:384:43" "all:21:42:43" "upper:192:384:44" "all:21:42:44")

{
  echo "=== alpha-matched 2x2 start $(date) ==="
  AVAIL=$(df -BG --output=avail "$DYS_ROOT" | tail -1 | tr -dc '0-9')
  echo "disk available: ${AVAIL} GB"
  if [ "${AVAIL:-0}" -lt 80 ]; then echo "FATAL: need >=80 GB"; echo disk > "$DONE"; exit 1; fi
  if [ ! -f "$MAN" ]; then echo "FATAL: manifest missing"; echo manifest > "$DONE"; exit 1; fi
  source venv_qwen3asr/bin/activate || { echo "FATAL venv"; echo venv > "$DONE"; exit 1; }

  fail=0
  for cell in "${CELLS[@]}"; do
    IFS=':' read -r v rk al sd <<< "$cell"
    tag="${v}_r${rk}_a${al}_s${sd}"
    if [ -f "$MARK/$tag.done" ]; then echo "SKIP  $tag (marker)"; continue; fi
    if python3 scripts/layerprobe/_am_have.py "$v" "$rk" "$al" "$sd"; then
      echo "SKIP  $tag (adapter already on disk at this alpha)"; date > "$MARK/$tag.done"; continue
    fi
    echo "########## TRAIN $tag @ $(date +%H:%M:%S) ##########"
    python -u scripts/layerprobe/train_lora_target_sweep.py \
        --variant "$v" --rank "$rk" --alpha "$al" --seed "$sd" --corpus cdsd --manifest "$MAN"
    rc=$?
    echo "$tag rc=$rc"
    if [ $rc -eq 0 ]; then date > "$MARK/$tag.done"; else fail=$((fail+1)); fi
  done
  echo "PHASE 1 complete, cells failed: $fail"

  for sd in 42 43 44; do
    OUT="results/probe/alpha_matched_s${sd}.csv"
    if [ -f "$OUT" ]; then echo "SKIP eval s$sd (exists)"; continue; fi
    mapfile -t DIRS < <(python3 scripts/layerprobe/_am_resolve.py "$sd")
    if [ "${#DIRS[@]}" -ne 4 ]; then
      echo "SKIP eval s$sd: resolved ${#DIRS[@]}/4 alpha-matched cells"
      continue
    fi
    source venv_qwen3asr/bin/activate
    ARGS=(--variant "base=Qwen/Qwen3-ASR-1.7B")
    ok=1
    for line in "${DIRS[@]}"; do
      b=$(printf '%s' "$line" | cut -f1)
      dir=$(printf '%s' "$line" | cut -f2)
      if [ ! -d "$dir/merged" ]; then
        # CPU-forced: peft pulls the adapter onto CUDA during merge and OOMs on this box's unified
        # memory once swap is engaged. Merging is weight arithmetic; it needs no accelerator.
        CUDA_VISIBLE_DEVICES="" python -u scripts/layerprobe/merge_adapter_export.py \
            --adapter "$dir/adapter" --out "$dir/merged"
        if [ $? -ne 0 ]; then echo "FATAL merge $b s$sd"; ok=0; break; fi
      fi
      ARGS+=(--variant "${b}_s${sd}=${dir}/merged")
    done
    if [ $ok -eq 0 ]; then continue; fi

    # The evaluator imports qwen_asr, which lives ONLY in the main venv, while the merge above needs
    # venv_qwen3asr. Sourcing one for both fails at decode time - after every merge is already paid for.
    deactivate 2>/dev/null || true
    source venv/bin/activate
    if ! python -c "import qwen_asr" 2>/dev/null; then
      echo "FATAL: qwen_asr missing in main venv"; continue
    fi
    echo "########## EVAL seed $sd @ $(date +%H:%M:%S) ##########"
    python -u scripts/layerprobe/eval_lora_variants.py \
        --manifest "$MAN" --split test \
        --out "$OUT" --preds-dir "results/probe/alpha_matched_preds/s${sd}" "${ARGS[@]}"
    echo "eval s$sd rc=$?"
    deactivate 2>/dev/null || true
  done

  echo "$fail" > "$DONE"
  echo "=== ALPHA-MATCHED 2x2 DONE $(date) ==="
} >> "$LOG" 2>&1
