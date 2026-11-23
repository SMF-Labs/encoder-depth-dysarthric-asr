#!/usr/bin/env bash
# Multi-seed replication of the CDSD x Qwen3-ASR band sweep (Paper 10a, reviewer's #1 request).
#
# WHY. The CDSD placement conclusions rest on ONE optimisation seed. The speaker bootstrap resamples the
# six held-out speakers; it says nothing about training stochasticity. The two headline contrasts are
# small - upper-middle = -1.52 pp, upper-all = -1.47 pp - so those intervals alone cannot call them
# reliable. This trains the four bands at rank 64 across seeds 42/43/44 so both contrasts get a seed
# distribution, matching the discipline already used on UASpeech.
#
# ORDER IS DELIBERATE. The full grid is ~35 GPU-hours, more than one night, so cells are ordered so the
# most decision-relevant land first:
#   cells 1-5   complete `upper` and `middle` at three seeds -> fixes the weakest claim (upper-middle has
#               no independent support elsewhere in the paper)
#   cells 6-7   complete `all`                              -> upper-all is already corroborated by the 2x2
#   cells 8-10  complete `lower`                            -> no claim depends on it
# Every cell is marker-guarded, so this can be stopped and relaunched and resumes at the first unfinished
# cell. `upper` and `all` at seed 42 already exist from the parameter-matched 2x2 and are reused.
#
# Scoring is NOT per cell: all variants for a seed are scored in ONE invocation so they share a decode
# path and manifest. Scoring separately would let a configuration difference masquerade as a result.
set -uo pipefail
# Repository root of the working tree these runs read and write. Override to relocate.
DYS_ROOT="${DYS_ROOT:?set DYS_ROOT to the dysarthria working tree, e.g. export DYS_ROOT=$HOME/dysarthria}"
cd "$DYS_ROOT" || exit 1

MAN=results/probe/cdsd_lora_manifest.jsonl
LOG=logs/cdsd_multiseed.log
DONE=logs/cdsd_multiseed.DONE
MARK=logs/cdsd_multiseed_markers
RANK=64
mkdir -p "$MARK"
rm -f "$DONE"

CELLS=("middle:42" "middle:43" "upper:43" "middle:44" "upper:44" "all:43" "all:44" "lower:42" "lower:43" "lower:44")

{
  echo "=== CDSD x Qwen multi-seed sweep start $(date) ==="
  AVAIL=$(df -BG --output=avail "$DYS_ROOT" | tail -1 | tr -dc '0-9')
  echo "disk available: ${AVAIL} GB"
  if [ "${AVAIL:-0}" -lt 120 ]; then echo "FATAL: need >=120 GB"; echo disk > "$DONE"; exit 1; fi
  if [ ! -f "$MAN" ]; then echo "FATAL: manifest missing"; echo manifest > "$DONE"; exit 1; fi
  source venv_qwen3asr/bin/activate || { echo "FATAL venv"; echo venv > "$DONE"; exit 1; }

  fail=0
  for cell in "${CELLS[@]}"; do
    v="${cell%%:*}"; sd="${cell##*:}"
    tag="${v}_r${RANK}_s${sd}"
    if [ -f "$MARK/$tag.done" ]; then echo "SKIP  $tag (marker)"; continue; fi
    if python3 scripts/layerprobe/_ms_have_adapter.py "$v" "$sd" "$RANK"; then
      echo "SKIP  $tag (adapter already on disk)"; date > "$MARK/$tag.done"; continue
    fi
    echo "########## TRAIN $tag @ $(date +%H:%M:%S) ##########"
    python -u scripts/layerprobe/train_lora_target_sweep.py \
        --variant "$v" --rank "$RANK" --seed "$sd" --corpus cdsd --manifest "$MAN"
    rc=$?
    echo "$tag rc=$rc"
    if [ $rc -eq 0 ]; then date > "$MARK/$tag.done"; else fail=$((fail+1)); fi
  done
  echo "PHASE 1 complete, cells failed: $fail"

  for sd in 42 43 44; do
    OUT="results/probe/cdsd_multiseed_s${sd}.csv"
    if [ -f "$OUT" ]; then echo "SKIP eval s$sd (exists)"; continue; fi
    mapfile -t DIRS < <(python3 scripts/layerprobe/_ms_resolve.py "$sd" "$RANK")
    if [ "${#DIRS[@]}" -ne 4 ]; then
      echo "SKIP eval s$sd: resolved ${#DIRS[@]}/4 adapters (training incomplete)"
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
        --out "$OUT" --preds-dir "results/probe/cdsd_multiseed_preds/s${sd}" "${ARGS[@]}"
    echo "eval s$sd rc=$?"
    deactivate 2>/dev/null || true
  done

  echo "$fail" > "$DONE"
  echo "=== CDSD MULTISEED DONE $(date) ==="
} >> "$LOG" 2>&1
