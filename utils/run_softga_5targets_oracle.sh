#!/usr/bin/env bash
set -euo pipefail

# Batch run SoftGA oracle-budget benchmark for five targets:
# parp1, fa7, 5ht1b, braf, jak2
#
# Default usage:
#   bash utils/run_softga_5targets_oracle.sh
#
# Optional environment overrides:
#   CONFIG=config.yaml
#   MAX_CALLS=1000
#   SEED=0
#   NUM_PROCESSORS=120
#   DOCKING_TOOL=qvina02
#   EXHAUSTIVENESS=8
#   SMI_FILE=utils/initial_population/initial_population.smi
#   BASE_OUTPUT_DIR=output/softga_5targets_seed0_calls1000
#   CLEANUP_INTERMEDIATE_FILES=0
#   DRY_RUN=1
# ---------------------------------
# bash utils/run_softga_5targets_oracle.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-config.yaml}"
MAX_CALLS="${MAX_CALLS:-1000}"
SEED="${SEED:-0}"
NUM_PROCESSORS="${NUM_PROCESSORS:-$(nproc)}"
DOCKING_TOOL="${DOCKING_TOOL:-qvina02}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-8}"
SMI_FILE="${SMI_FILE:-utils/initial_population/initial_population.smi}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-output/softga_5targets_seed${SEED}_calls${MAX_CALLS}}"
CLEANUP_INTERMEDIATE_FILES="${CLEANUP_INTERMEDIATE_FILES:-0}"
DRY_RUN="${DRY_RUN:-0}"

TARGETS=(parp1 fa7 5ht1b braf jak2)

run_target() {
  local target="$1"
  local run_dir="${BASE_OUTPUT_DIR}/${target}"
  local success_csv="oracle_calls_success.csv"

  local cmd=(
    python main.py
    --config "${CONFIG}"
    --receptor "${target}"
    --output_dir "${BASE_OUTPUT_DIR}"
    --initial_population_file "${SMI_FILE}"
    --seed "${SEED}"
    --number_of_processors "${NUM_PROCESSORS}"
    --docking_tool "${DOCKING_TOOL}"
    --docking_exhaustiveness "${EXHAUSTIVENESS}"
    --max_oracle_calls "${MAX_CALLS}"
  )

  if [[ "${CLEANUP_INTERMEDIATE_FILES}" == "1" ]]; then
    cmd+=(--cleanup_intermediate_files true)
  fi

  echo "============================================================"
  echo "Target: ${target}"
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

failed=()
for t in "${TARGETS[@]}"; do
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
