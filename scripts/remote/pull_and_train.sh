#!/usr/bin/env bash
set -euo pipefail

# Pull latest code from GitHub (HTTPS+PAT auth must be configured in git credentials)
# and start training with GPU guard.

PROJECT_DIR="/home/jiajie/yhong/lsw/project"
BRANCH="main"
SESSION_NAME="train_full"
TRAIN_CMD="python -m src.train --model full --config configs/train.yaml --override paths.data_root=/home/jiajie/yhong/lsw/data --override paths.processed_dir=/home/jiajie/yhong/lsw/artifacts/processed --override output.runs_dir=/home/jiajie/yhong/lsw/runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --session) SESSION_NAME="$2"; shift 2;;
    --train-cmd) TRAIN_CMD="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

cd "$PROJECT_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

python -m src.prepare_data --config configs/data.yaml --override paths.data_root=/home/jiajie/yhong/lsw/data --override paths.processed_dir=/home/jiajie/yhong/lsw/artifacts/processed --override output.bp_index=/home/jiajie/yhong/lsw/artifacts/processed/bp_index.csv --override output.signal_index=/home/jiajie/yhong/lsw/artifacts/processed/signal_index.csv --override output.sample_index=/home/jiajie/yhong/lsw/artifacts/processed/sample_index.csv
python -m src.make_splits --config configs/split.yaml --override input.sample_index=/home/jiajie/yhong/lsw/artifacts/processed/sample_index.csv --override output.split_dir=/home/jiajie/yhong/lsw/artifacts/processed/splits

bash scripts/remote/run_with_gpu_guard.sh --session "$SESSION_NAME" --command "$TRAIN_CMD"
