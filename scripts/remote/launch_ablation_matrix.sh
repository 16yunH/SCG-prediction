#!/usr/bin/env bash
set -euo pipefail

# Queue 4-model ablation jobs across a GPU pool. Each tmux worker owns one GPU and
# runs assigned fold/final jobs sequentially with live stdout plus log persistence.

CONDA_ENV="yh"
PROJECT_DIR="$HOME/yhong/lsw/project"
SPLIT_DIR="/home/jiajie/yhong/lsw/artifacts/processed/v2/splits"
RUNS_DIR="/home/jiajie/yhong/lsw/runs"
LOG_DIR="/home/jiajie/yhong/lsw/runs/logs"
GPU_POOL="1,2,3,5,6,7"
FOLDS="5"
MODELS="full,cnn_only,lstm_only,mlp_only"
WORKERS="4"
BATCH_SIZE="512"
EPOCHS="30"
INCLUDE_FINAL="true"
GPU_UTIL_THRESHOLD="30"
GPU_MEM_THRESHOLD="50"
SESSION_PREFIX="ablation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --project-dir) PROJECT_DIR="$2"; shift 2;;
    --split-dir) SPLIT_DIR="$2"; shift 2;;
    --runs-dir) RUNS_DIR="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --gpu-pool) GPU_POOL="$2"; shift 2;;
    --folds) FOLDS="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --include-final) INCLUDE_FINAL="$2"; shift 2;;
    --gpu-util-threshold) GPU_UTIL_THRESHOLD="$2"; shift 2;;
    --gpu-mem-threshold) GPU_MEM_THRESHOLD="$2"; shift 2;;
    --session-prefix) SESSION_PREFIX="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

mkdir -p "$LOG_DIR" "$RUNS_DIR"

is_gpu_available() {
  local target="$1"
  local line idx util mem_used mem_total mem_pct
  line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' -v g="$target" '$1+0==g+0 {print $0}')
  [[ -n "$line" ]] || return 1
  IFS=',' read -r idx util mem_used mem_total <<< "$line"
  util=$(echo "$util" | xargs)
  mem_used=$(echo "$mem_used" | xargs)
  mem_total=$(echo "$mem_total" | xargs)
  mem_pct=$(( mem_used * 100 / mem_total ))
  [[ "$util" -lt "$GPU_UTIL_THRESHOLD" && "$mem_pct" -lt "$GPU_MEM_THRESHOLD" ]]
}

IFS=',' read -r -a requested_gpus <<< "$GPU_POOL"
available_gpus=()
if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu in "${requested_gpus[@]}"; do
    gpu=$(echo "$gpu" | xargs)
    if is_gpu_available "$gpu"; then
      available_gpus+=("$gpu")
    else
      echo "[warn] skip busy GPU=$gpu"
    fi
  done
else
  echo "[warn] nvidia-smi not found; using requested pool without filtering"
  available_gpus=("${requested_gpus[@]}")
fi

if [[ "${#available_gpus[@]}" -eq 0 ]]; then
  echo "No GPU passed thresholds; refusing to start. Adjust --gpu-pool or thresholds." >&2
  exit 1
fi

IFS=',' read -r -a model_list <<< "$MODELS"
tasks=()
for model in "${model_list[@]}"; do
  model=$(echo "$model" | xargs)
  for fold in $(seq 1 "$FOLDS"); do
    tasks+=("${model}:cv:${fold}")
  done
  if [[ "$INCLUDE_FINAL" == "true" ]]; then
    tasks+=("${model}:final:0")
  fi
done

run_task_cmd() {
  local model="$1"
  local mode="$2"
  local fold="$3"
  local extra=""
  if [[ "$mode" == "cv" ]]; then
    extra="--mode cv --fold ${fold}"
  else
    extra="--mode final"
  fi
  cat <<EOF
python -u -m src.train --model ${model} --config configs/train.yaml ${extra} \
  --override input.split_dir=${SPLIT_DIR} \
  --override output.runs_dir=${RUNS_DIR} \
  --override runtime.num_workers=${WORKERS} \
  --override optimization.batch_size=${BATCH_SIZE} \
  --override optimization.epochs=${EPOCHS} \
  --override runtime.pin_memory=true \
  --override runtime.cache_windows=true \
  --override runtime.mmap_arrays=true
EOF
}

for i in "${!available_gpus[@]}"; do
  gpu="${available_gpus[$i]}"
  session="${SESSION_PREFIX}_gpu${gpu}"
  tmux kill-session -t "$session" 2>/dev/null || true
  worker_file=$(mktemp)
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$PROJECT_DIR'"
    echo "source ~/anaconda3/etc/profile.d/conda.sh"
    echo "conda activate '$CONDA_ENV'"
    echo "export CUDA_VISIBLE_DEVICES='$gpu'"
    echo "export PYTHONUNBUFFERED=1"
    echo "echo '[worker] session=$session physical_gpu=$gpu visible_gpu=0'"
    for j in "${!tasks[@]}"; do
      if (( j % ${#available_gpus[@]} == i )); then
        task="${tasks[$j]}"
        IFS=':' read -r model mode fold <<< "$task"
        log_file="${LOG_DIR}/${SESSION_PREFIX}_${model}_${mode}_${fold}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
        echo "echo '[task-start] model=$model mode=$mode fold=$fold gpu=$gpu log=$log_file'"
        echo "$(run_task_cmd "$model" "$mode" "$fold") 2>&1 | tee -a '$log_file'"
        echo "echo '[task-done] model=$model mode=$mode fold=$fold gpu=$gpu'"
      fi
    done
    echo "echo '[worker-done] session=$session gpu=$gpu'"
  } > "$worker_file"
  chmod +x "$worker_file"
  tmux new-session -d -s "$session" "bash '$worker_file'"
  echo "started $session on GPU $gpu"
done

echo "Started ${#tasks[@]} tasks across ${#available_gpus[@]} GPUs: ${available_gpus[*]}"
echo "Attach a worker: tmux attach -t ${SESSION_PREFIX}_gpu${available_gpus[0]}"
echo "List sessions: tmux ls"
