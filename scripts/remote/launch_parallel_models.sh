#!/usr/bin/env bash
set -euo pipefail

# Launch 4 model variants in parallel on fixed GPUs for higher utilization and predictable placement.
# Example:
# bash scripts/remote/launch_parallel_models.sh --conda-env yh --split-dir /home/jiajie/yhong/lsw/artifacts/processed/splits --runs-dir /home/jiajie/yhong/lsw/runs --workers 12 --batch-size 512

CONDA_ENV="yh"
SPLIT_DIR="/home/jiajie/yhong/lsw/artifacts/processed/splits"
RUNS_DIR="/home/jiajie/yhong/lsw/runs"
WORKERS="12"
BATCH_SIZE="512"

# fixed distinct GPUs (change if needed)
GPU_FULL="1"
GPU_CNN="2"
GPU_LSTM="3"
GPU_MLP="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --split-dir) SPLIT_DIR="$2"; shift 2;;
    --runs-dir) RUNS_DIR="$2"; shift 2;;
    --workers) WORKERS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --gpu-full) GPU_FULL="$2"; shift 2;;
    --gpu-cnn) GPU_CNN="$2"; shift 2;;
    --gpu-lstm) GPU_LSTM="$2"; shift 2;;
    --gpu-mlp) GPU_MLP="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

mkdir -p /home/jiajie/yhong/lsw/runs/logs

launch_one() {
  local model="$1"
  local session="$2"
  local gpu="$3"

  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd ~/yhong/lsw/project; CUDA_VISIBLE_DEVICES=${gpu} conda run -n ${CONDA_ENV} --no-capture-output python -u -m src.train --model ${model} --config configs/train.yaml --override input.split_dir=${SPLIT_DIR} --override output.runs_dir=${RUNS_DIR} --override runtime.num_workers=${WORKERS} --override optimization.batch_size=${BATCH_SIZE} --override runtime.pin_memory=true --override runtime.cache_windows=true > /home/jiajie/yhong/lsw/runs/logs/${session}.log 2>&1"

  echo "started ${session} on GPU ${gpu}"
}

launch_one full train_full "$GPU_FULL"
launch_one cnn_only train_cnn "$GPU_CNN"
launch_one lstm_only train_lstm "$GPU_LSTM"
launch_one mlp_only train_mlp "$GPU_MLP"

echo "All sessions started."
tmux ls
