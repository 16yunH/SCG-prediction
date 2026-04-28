#!/usr/bin/env bash
set -euo pipefail

# Launch multiple model variants in parallel to utilize multiple GPUs.
# Example:
# bash scripts/remote/launch_parallel_models.sh --conda-env yh --split-dir /home/jiajie/yhong/lsw/artifacts/processed/splits --runs-dir /home/jiajie/yhong/lsw/runs

CONDA_ENV="yh"
SPLIT_DIR="/home/jiajie/yhong/lsw/artifacts/processed/splits"
RUNS_DIR="/home/jiajie/yhong/lsw/runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --split-dir) SPLIT_DIR="$2"; shift 2;;
    --runs-dir) RUNS_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

chmod +x scripts/remote/run_with_gpu_guard.sh

launch_one() {
  local model="$1"
  local session="$2"
  local gpu_ids="$3"

  bash scripts/remote/run_with_gpu_guard.sh \
    --gpu-util-threshold 30 \
    --gpu-mem-threshold 10 \
    --poll-seconds 30 \
    --session "$session" \
    --command "source ~/anaconda3/etc/profile.d/conda.sh; conda activate ${CONDA_ENV}; python -u -m src.train --model ${model} --config configs/train.yaml --override input.split_dir=${SPLIT_DIR} --override output.runs_dir=${RUNS_DIR} --override runtime.gpu_ids=${gpu_ids} --override runtime.use_data_parallel=false"
}

# Spread models on distinct GPUs by pinning CUDA_VISIBLE_DEVICES via each session's guard selection.
launch_one full train_full 1
launch_one cnn_only train_cnn 2
launch_one lstm_only train_lstm 3
launch_one mlp_only train_mlp 5

echo "Started: train_full, train_cnn, train_lstm, train_mlp"
