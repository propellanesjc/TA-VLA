#!/usr/bin/env bash

set -euo pipefail

MODEL_CONFIG=""
CHECKPOINT_DIR=""
HOST="127.0.0.1"
PORT=8000
NUM_TRIALS=10
SEED=7
OUTPUT_DIR="./batch_results/tavla_vla_arena"
SKIP_EXISTING=false
START_SERVER=false
USE_EFFORT=true
EFFORT_HISTORY_LEN=1
EFFORT_DIM=7

TASK_SUITES=(
  "safety_dynamic_obstacles"
  "safety_hazard_avoidance"
  "safety_state_preservation"
  "safety_cautious_grasp"
  "safety_static_obstacles"
  "distractor_dynamic_distractors"
  "distractor_static_distractors"
  "extrapolation_preposition_combinations"
  "extrapolation_task_workflows"
  "extrapolation_unseen_objects"
  "long_horizon"
)
TASK_LEVELS=(0 1 2)

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS]

Batch evaluate TA-VLA server-client on VLA-Arena.

OPTIONS:
  --model-config NAME       Config name for serve_policy (e.g. pi0_lora_effort)
  --checkpoint-dir PATH     Checkpoint dir for serve_policy
  --host HOST               Policy server host (default: ${HOST})
  --port PORT               Policy server port (default: ${PORT})
  --trials NUM              Trials per task (default: ${NUM_TRIALS})
  --seed NUM                Seed (default: ${SEED})
  --output-dir DIR          Output directory (default: ${OUTPUT_DIR})
  --suites "a b"            Space-separated suites override
  --levels "0 1 2"          Space-separated levels override
  --skip-existing           Skip run if expected log already exists
  --start-server            Auto start local TA-VLA websocket server
  --use-effort true|false   Whether to attach observation/effort (default: true)
  --effort-history-len NUM  Effort history length passed to policy (default: 1)
  --effort-dim NUM          Effort dimension (default: 7)
  -h, --help                Show this help
EOF
}

CUSTOM_SUITES=""
CUSTOM_LEVELS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --trials) NUM_TRIALS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --suites) CUSTOM_SUITES="$2"; shift 2 ;;
    --levels) CUSTOM_LEVELS="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=true; shift ;;
    --start-server) START_SERVER=true; shift ;;
    --use-effort) USE_EFFORT="$2"; shift 2 ;;
    --effort-history-len) EFFORT_HISTORY_LEN="$2"; shift 2 ;;
    --effort-dim) EFFORT_DIM="$2"; shift 2 ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; show_usage; exit 1 ;;
  esac
done

if [[ -n "${CUSTOM_SUITES}" ]]; then
  # shellcheck disable=SC2206
  TASK_SUITES=(${CUSTOM_SUITES})
fi
if [[ -n "${CUSTOM_LEVELS}" ]]; then
  # shellcheck disable=SC2206
  TASK_LEVELS=(${CUSTOM_LEVELS})
fi

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="${OUTPUT_DIR}/batch_eval_summary_${TIMESTAMP}.csv"
echo "Task Suite,Level,Success Rate,Successes,Total Episodes,Average Cost,Success Costs,Failure Costs,Log File" > "${SUMMARY_FILE}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    echo "[INFO] Stopping server PID=${SERVER_PID}"
    kill "${SERVER_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${START_SERVER}" == "true" ]]; then
  if [[ -z "${MODEL_CONFIG}" || -z "${CHECKPOINT_DIR}" ]]; then
    echo "[ERROR] --start-server requires --model-config and --checkpoint-dir"
    exit 1
  fi

  echo "[INFO] Starting TA-VLA server on port ${PORT} ..."
  uv run scripts/serve_policy.py \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${MODEL_CONFIG}" \
    --policy.dir "${CHECKPOINT_DIR}" \
    > "${OUTPUT_DIR}/server_${TIMESTAMP}.log" 2>&1 &
  SERVER_PID=$!

  for _ in {1..60}; do
    if nc -z "${HOST}" "${PORT}" >/dev/null 2>&1; then
      echo "[INFO] Server is ready at ${HOST}:${PORT}"
      break
    fi
    sleep 1
  done
fi

extract_success_rate() {
  local log_file="$1"
  local v
  v=$(grep -i "Total success rate:" "${log_file}" | tail -1 | sed 's/.*Total success rate: //I' | awk '{print $1}' | tr -d '%')
  echo "${v:-N/A}"
}

extract_total_episodes() {
  local log_file="$1"
  local v
  v=$(grep -E -i "# episodes completed so far:|Total episodes:" "${log_file}" | tail -1 | sed 's/.*: //' | awk '{print $1}')
  echo "${v:-N/A}"
}

extract_total_successes() {
  local log_file="$1"
  local v
  v=$(grep -E -i "# successes:|Total successes:" "${log_file}" | tail -1 | sed 's/.*: //' | awk '{print $1}')
  echo "${v:-N/A}"
}

extract_average_cost() {
  local log_file="$1"
  local v
  v=$(grep -i "Average cost:" "${log_file}" | tail -1 | sed 's/.*Average cost: //I' | awk '{print $1}')
  echo "${v:-N/A}"
}

extract_success_costs() {
  local log_file="$1"
  local v
  v=$(grep -i "Success costs:" "${log_file}" | tail -1 | sed 's/.*Success costs: //I' | awk '{print $1}')
  echo "${v:-N/A}"
}

extract_failure_costs() {
  local log_file="$1"
  local v
  v=$(grep -i "Failure costs:" "${log_file}" | tail -1 | sed 's/.*Failure costs: //I' | awk '{print $1}')
  echo "${v:-N/A}"
}

total_evals=$((${#TASK_SUITES[@]} * ${#TASK_LEVELS[@]}))
current_eval=0
ok_count=0
fail_count=0

for suite in "${TASK_SUITES[@]}"; do
  for level in "${TASK_LEVELS[@]}"; do
    current_eval=$((current_eval + 1))
    run_id="EVAL-${suite}-L${level}-${TIMESTAMP}"
    log_file="${OUTPUT_DIR}/${run_id}.txt"

    echo "[INFO] Progress ${current_eval}/${total_evals} | suite=${suite}, level=${level}"

    if [[ "${SKIP_EXISTING}" == "true" && -f "${log_file}" ]]; then
      echo "[INFO] Skip existing log: ${log_file}"
      continue
    fi

    cmd=(
      uv run examples/vla_arena/main.py
      --vla-arena-root ../VLA-Arena
      --host "${HOST}"
      --port "${PORT}"
      --task-suite-name "${suite}"
      --task-level "${level}"
      --num-trials-per-task "${NUM_TRIALS}"
      --seed "${SEED}"
      --save-video-mode none
      --local-log-dir "${OUTPUT_DIR}"
      --effort-history-len "${EFFORT_HISTORY_LEN}"
      --effort-dim "${EFFORT_DIM}"
    )

    if [[ "${USE_EFFORT}" == "false" ]]; then
      cmd+=(--no-use-effort)
    else
      cmd+=(--use-effort)
    fi

    if "${cmd[@]}" > "${log_file}" 2>&1; then
      sr=$(extract_success_rate "${log_file}")
      te=$(extract_total_episodes "${log_file}")
      ts=$(extract_total_successes "${log_file}")
      ac=$(extract_average_cost "${log_file}")
      sc=$(extract_success_costs "${log_file}")
      fc=$(extract_failure_costs "${log_file}")
      echo "[SUCCESS] ${suite} L${level}: SR=${sr}% (${ts}/${te}), AvgCost=${ac}"
      echo "${suite},L${level},${sr},${ts},${te},${ac},${sc},${fc},${log_file}" >> "${SUMMARY_FILE}"
      ok_count=$((ok_count + 1))
    else
      echo "[ERROR] Failed ${suite} L${level}; tail log:"
      tail -n 40 "${log_file}" || true
      echo "${suite},L${level},FAILED,N/A,N/A,N/A,N/A,N/A,${log_file}" >> "${SUMMARY_FILE}"
      fail_count=$((fail_count + 1))
    fi

    sleep 1
  done
done

echo "[INFO] Batch done. success=${ok_count}, failed=${fail_count}"
echo "[INFO] Summary: ${SUMMARY_FILE}"
