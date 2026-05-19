#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# bash scripts/remote/run_with_gpu_guard.sh \
#   --gpu-util-threshold 30 \
#   --gpu-mem-threshold 50 \
#   --poll-seconds 60 \
#   --session train_full \
#   --command "python -m src train --model full --config configs/train.yaml"

GPU_UTIL_THRESHOLD=30
GPU_MEM_THRESHOLD=50
POLL_SECONDS=60
SESSION_NAME="train_job"
RUN_COMMAND=""
LOG_DIR="/home/jiajie/yhong/lsw/runs/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-util-threshold)
      GPU_UTIL_THRESHOLD="$2"; shift 2;;
    --gpu-mem-threshold)
      GPU_MEM_THRESHOLD="$2"; shift 2;;
    --poll-seconds)
      POLL_SECONDS="$2"; shift 2;;
    --session)
      SESSION_NAME="$2"; shift 2;;
    --command)
      RUN_COMMAND="$2"; shift 2;;
    --log-dir)
      LOG_DIR="$2"; shift 2;;
    *)
      echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$RUN_COMMAND" ]]; then
  echo "--command is required" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

pick_gpu() {
  local picked=""
  while IFS=',' read -r idx util mem_used mem_total; do
    idx=$(echo "$idx" | xargs)
    util=$(echo "$util" | xargs)
    mem_used=$(echo "$mem_used" | xargs)
    mem_total=$(echo "$mem_total" | xargs)

    if [[ -z "$idx" || -z "$util" || -z "$mem_used" || -z "$mem_total" ]]; then
      continue
    fi

    if [[ "$mem_total" -le 0 ]]; then
      continue
    fi

    local mem_pct=$(( mem_used * 100 / mem_total ))
    echo "[$(date '+%F %T')] GPU=$idx util=${util}% mem=${mem_used}/${mem_total}MB (${mem_pct}%)" | tee -a "$LOG_FILE" >&2

    if [[ "$util" -lt "$GPU_UTIL_THRESHOLD" && "$mem_pct" -lt "$GPU_MEM_THRESHOLD" ]]; then
      picked="$idx"
      break
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)

  echo "$picked"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" | tee -a "$LOG_FILE"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found" | tee -a "$LOG_FILE"
  exit 1
fi

SELECTED_GPU=""
while [[ -z "$SELECTED_GPU" ]]; do
  SELECTED_GPU=$(pick_gpu)
  if [[ -z "$SELECTED_GPU" ]]; then
    echo "[$(date '+%F %T')] No GPU available. Sleep ${POLL_SECONDS}s" | tee -a "$LOG_FILE"
    sleep "$POLL_SECONDS"
  fi
done

echo "[$(date '+%F %T')] Selected GPU=$SELECTED_GPU" | tee -a "$LOG_FILE"

# Unbuffered Python output and line-buffered pipes for live tmux visibility.
TMUX_CMD="export CUDA_VISIBLE_DEVICES=${SELECTED_GPU}; export PYTHONUNBUFFERED=1; stdbuf -oL -eL bash -lc '${RUN_COMMAND}' 2>&1 | tee -a ${LOG_FILE}"

# Replace existing session if present.
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

TMUX_TMP=$(mktemp)
printf '%s\n' "$TMUX_CMD" > "$TMUX_TMP"
tmux new-session -d -s "$SESSION_NAME" "bash $TMUX_TMP"

echo "Session started: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION_NAME"
