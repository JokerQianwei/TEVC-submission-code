#!/usr/bin/env bash
set -euo pipefail

# Serial recommended 5-target run under 1000 success-oracle budget.
# Targets: parp1 fa7 5ht1b braf jak2
# Note: alias "5htlb" will be mapped to "5ht1b".
#
# Usage:
#   bash utils/run_softga_5targets_recommended.sh
#
# Optional overrides:
#   CONFIG=config.yaml
#   TARGETS="parp1 fa7 5ht1b braf jak2"
#   MAX_CALLS=1000
#   SEED=0
#   NUM_PROCESSORS=120
#   DOCKING_TOOL=qvina02
#   EXHAUSTIVENESS=8
#   SMI_FILE=utils/initial_population/initial_population.smi
#   BASE_OUTPUT_DIR=output/softga_5targets_recommended_seed0_calls1000
#   CLEANUP_INTERMEDIATE_FILES=0
#   DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-config.yaml}"
TARGETS="${TARGETS:-parp1 fa7 5ht1b braf jak2}"
MAX_CALLS="${MAX_CALLS:-1000}"
SEED="${SEED:-0}"
NUM_PROCESSORS="${NUM_PROCESSORS:-120}"
DOCKING_TOOL="${DOCKING_TOOL:-qvina02}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-8}"
SMI_FILE="${SMI_FILE:-utils/initial_population/initial_population.smi}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-output/softga_5targets_recommended_seed${SEED}_calls${MAX_CALLS}}"
CLEANUP_INTERMEDIATE_FILES="${CLEANUP_INTERMEDIATE_FILES:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Recommended parameter set (can be overridden by env if needed).
STRATEGY_MODE="${STRATEGY_MODE:-cosine}"
MIN_KEEP_RATIO="${MIN_KEEP_RATIO:-0.05}"
MAX_KEEP_RATIO="${MAX_KEEP_RATIO:-0.88}"
GEN1_N_SELECT="${GEN1_N_SELECT:-80}"
GEN1_SELECTION_MODE="${GEN1_SELECTION_MODE:-maxmin}"
SAMPLES_PER_PARENT="${SAMPLES_PER_PARENT:-16}"
TANIMOTO_THRESHOLD="${TANIMOTO_THRESHOLD:-0.58}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUCLEUS_P="${NUCLEUS_P:-0.9}"
STEPS="${STEPS:-80}"
RECIRCLE="${RECIRCLE:-True}"
NUMBER_OF_CROSSOVERS="${NUMBER_OF_CROSSOVERS:-40}"
NUMBER_OF_MUTANTS="${NUMBER_OF_MUTANTS:-130}"
PAIR_MIN_TANIMOTO="${PAIR_MIN_TANIMOTO:-0.3}"
PAIR_MAX_TANIMOTO="${PAIR_MAX_TANIMOTO:-0.75}"
MAX_GENERATIONS="${MAX_GENERATIONS:-30}"

canonical_target() {
  local t="$1"
  case "${t}" in
    5htlb) echo "5ht1b" ;;
    *) echo "${t}" ;;
  esac
}

run_target() {
  local raw_target="$1"
  local target
  target="$(canonical_target "${raw_target}")"
  local run_dir="${BASE_OUTPUT_DIR}/${target}"
  local success_csv="oracle_calls_success.csv"

  local cmd=(
    python main.py
    --config "${CONFIG}"
    --receptor "${target}"
    --output_dir "${BASE_OUTPUT_DIR}"
    --initial_population_file "${SMI_FILE}"
    --seed "${SEED}"
    --softbd_seed_mode workflow
    --number_of_processors "${NUM_PROCESSORS}"
    --docking_tool "${DOCKING_TOOL}"
    --docking_exhaustiveness "${EXHAUSTIVENESS}"
    --max_oracle_calls "${MAX_CALLS}"
    --plot_top1 false
    --strategy_mode "${STRATEGY_MODE}"
    --min_keep_ratio "${MIN_KEEP_RATIO}"
    --max_keep_ratio "${MAX_KEEP_RATIO}"
    --gen1_n_select "${GEN1_N_SELECT}"
    --gen1_selection_mode "${GEN1_SELECTION_MODE}"
    --samples_per_parent "${SAMPLES_PER_PARENT}"
    --tanimoto_threshold "${TANIMOTO_THRESHOLD}"
    --temperature "${TEMPERATURE}"
    --nucleus_p "${NUCLEUS_P}"
    --steps "${STEPS}"
    --recircle "${RECIRCLE}"
    --number_of_crossovers "${NUMBER_OF_CROSSOVERS}"
    --number_of_mutants "${NUMBER_OF_MUTANTS}"
    --pair_min_tanimoto "${PAIR_MIN_TANIMOTO}"
    --pair_max_tanimoto "${PAIR_MAX_TANIMOTO}"
    --max_generations "${MAX_GENERATIONS}"
    --cleanup_intermediate_files false
  )

  if [[ "${CLEANUP_INTERMEDIATE_FILES}" == "1" ]]; then
    cmd+=(--cleanup_intermediate_files true)
  fi

  echo "============================================================"
  echo "Target(raw): ${raw_target}"
  echo "Target(run): ${target}"
  echo "Output: ${run_dir}"
  echo "Command: ${cmd[*]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if ! "${cmd[@]}"; then
    echo "[ERROR] target ${target} run failed"
    return 1
  fi

  local success_path="${run_dir}/${success_csv}"
  if [[ ! -f "${success_path}" ]]; then
    echo "[ERROR] missing oracle success csv: ${success_path}"
    return 1
  fi

  local success_count
  success_count=$(( $(wc -l < "${success_path}") - 1 ))
  if [[ "${success_count}" -ne "${MAX_CALLS}" ]]; then
    echo "[ERROR] target ${target} success oracle count mismatch: expected=${MAX_CALLS}, got=${success_count}"
    return 1
  fi

  echo "[OK] target ${target} finished with success oracle calls=${success_count}"
  return 0
}

read -r -a TARGET_ARR <<< "${TARGETS}"

failed=()
for t in "${TARGET_ARR[@]}"; do
  if ! run_target "${t}"; then
    failed+=("${t}")
  fi
done

echo "============================================================"
if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Finished with failures: ${failed[*]}"
  exit 1
fi

echo "All targets finished successfully."
echo "Base output directory: ${BASE_OUTPUT_DIR}"
