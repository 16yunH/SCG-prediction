#!/usr/bin/env bash
set -euo pipefail

# One-command v3 training launcher for a multi-GPU Linux server.
# It pulls the latest code, prepares v3 data/splits, then runs TCN final + 5 CV
# jobs in parallel across the available GPU pool using tmux.
#
# Example:
#   bash scripts/remote/train_v3_8gpu.sh \
#     --project-dir /home/jiajie/yhong/lsw/project \
#     --data-root /home/jiajie/yhong/lsw/data \
#     --conda-env lsw-bp \
#     --gpu-pool 0,1,2,3,4,5,6,7

REPO_URL="https://github.com/16yunH/SCG-prediction.git"
BRANCH="main"
PROJECT_DIR="$HOME/yhong/lsw/project"
DATA_ROOT="$HOME/yhong/lsw/data"
ARTIFACT_ROOT="$HOME/yhong/lsw/artifacts"
RUNS_DIR="$HOME/yhong/lsw/runs_v3"
LOG_DIR="$RUNS_DIR/logs"
CONDA_ENV="lsw-bp"
PYTHON_VERSION="3.10"
GPU_POOL="0,1,2,3,4,5,6,7"
MODEL="tcn"
SPLIT_KIND="calibrated"
BATCH_SIZE="64"
EPOCHS="200"
PATIENCE="30"
NUM_WORKERS="4"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
SKIP_PULL="false"
SKIP_INSTALL="false"
SKIP_PREPARE="false"
SESSION_PREFIX="v3_tcn"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url) REPO_URL="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --project-dir) PROJECT_DIR="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2;;
    --runs-dir) RUNS_DIR="$2"; LOG_DIR="$2/logs"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;
    --conda-env) CONDA_ENV="$2"; shift 2;;
    --python-version) PYTHON_VERSION="$2"; shift 2;;
    --gpu-pool) GPU_POOL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --split-kind) SPLIT_KIND="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2;;
    --skip-pull) SKIP_PULL="true"; shift;;
    --skip-install) SKIP_INSTALL="true"; shift;;
    --skip-prepare) SKIP_PREPARE="true"; shift;;
    --session-prefix) SESSION_PREFIX="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load Anaconda/Miniconda first." >&2
  exit 1
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux first." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script expects an NVIDIA GPU server." >&2
  exit 1
fi

mkdir -p "$(dirname "$PROJECT_DIR")" "$ARTIFACT_ROOT" "$RUNS_DIR" "$LOG_DIR"

if [[ -e "$PROJECT_DIR" && ! -d "$PROJECT_DIR/.git" ]]; then
  echo "$PROJECT_DIR exists but is not a git repository. Move it or choose another --project-dir." >&2
  exit 1
fi
if [[ ! -e "$PROJECT_DIR" ]]; then
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
if [[ "$SKIP_PULL" != "true" ]]; then
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}"
fi
conda activate "$CONDA_ENV"

if [[ "$SKIP_INSTALL" != "true" ]]; then
  python -m pip install --upgrade pip
  python -m pip install --upgrade --index-url "$TORCH_INDEX_URL" torch
  python -m pip install -e .
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment.")
PY

PROCESSED_DIR="$ARTIFACT_ROOT/processed/v3"
ARRAYS_DIR="$PROCESSED_DIR/arrays"
STD_SPLIT_DIR="$PROCESSED_DIR/splits"
CAL_SPLIT_DIR="$PROCESSED_DIR/calibrated_splits"
mkdir -p "$PROCESSED_DIR" "$ARRAYS_DIR"

if [[ "$SKIP_PREPARE" != "true" ]]; then
  python -m src prepare-data --config configs/data_v3.yaml \
    --override paths.data_root="$DATA_ROOT" \
    --override paths.artifacts_root="$ARTIFACT_ROOT" \
    --override paths.processed_dir="$PROCESSED_DIR" \
    --override paths.arrays_dir="$ARRAYS_DIR" \
    --override output.raw_manifest="$PROCESSED_DIR/raw_manifest.csv" \
    --override output.bp_index="$PROCESSED_DIR/bp_index.csv" \
    --override output.bp_corrections="$PROCESSED_DIR/bp_corrections.csv" \
    --override output.signal_index="$PROCESSED_DIR/signal_index.csv" \
    --override output.window_index="$PROCESSED_DIR/window_index.csv" \
    --override output.sample_index="$PROCESSED_DIR/sample_index.csv" \
    --override output.unlabeled_window_index="$PROCESSED_DIR/unlabeled_window_index.csv" \
    --override output.qc_report="$PROCESSED_DIR/qc_report.json"

  python -m src make-splits --config configs/split_v3.yaml \
    --override input.sample_index="$PROCESSED_DIR/window_index.csv" \
    --override output.split_dir="$STD_SPLIT_DIR"

  python -m src make-calibrated-splits --config configs/calibrated_split_v3.yaml \
    --override input.sample_index="$PROCESSED_DIR/window_index.csv" \
    --override output.split_dir="$CAL_SPLIT_DIR"
fi

case "$SPLIT_KIND" in
  calibrated)
    SPLIT_DIR="$CAL_SPLIT_DIR"
    ALLOW_SUBJECT_OVERLAP="true"
    ;;
  subject|subject-independent)
    SPLIT_DIR="$STD_SPLIT_DIR"
    ALLOW_SUBJECT_OVERLAP="false"
    ;;
  *)
    echo "Unknown --split-kind: $SPLIT_KIND (use calibrated or subject)" >&2
    exit 1
    ;;
esac

IFS=',' read -r -a GPUS <<< "$GPU_POOL"
tasks=("final:0")
for fold in 1 2 3 4 5; do
  tasks+=("cv:${fold}")
done

echo "[launch] project=$PROJECT_DIR"
echo "[launch] data=$DATA_ROOT"
echo "[launch] split=$SPLIT_DIR"
echo "[launch] runs=$RUNS_DIR"
echo "[launch] logs=$LOG_DIR"
echo "[launch] tasks=${tasks[*]}"
echo "[launch] gpus=${GPUS[*]}"

for i in "${!tasks[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  IFS=':' read -r mode fold <<< "${tasks[$i]}"
  session="${SESSION_PREFIX}_${SPLIT_KIND}_${mode}${fold}_gpu${gpu}"
  tmux kill-session -t "$session" 2>/dev/null || true
  task_log="$LOG_DIR/${session}_$(date +%Y%m%d_%H%M%S).log"
  task_script="$LOG_DIR/${session}.sh"

  mode_args=(--mode "$mode")
  if [[ "$mode" == "cv" ]]; then
    mode_args+=(--fold "$fold")
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$PROJECT_DIR'"
    echo "source \"\$(conda info --base)/etc/profile.d/conda.sh\""
    echo "conda activate '$CONDA_ENV'"
    echo "export CUDA_VISIBLE_DEVICES='$gpu'"
    echo "export PYTHONUNBUFFERED=1"
    echo "echo '[task-start] session=$session physical_gpu=$gpu mode=$mode fold=$fold'"
    printf "python -u -m src train --model '%s' --config configs/train_v3.yaml" "$MODEL"
    printf " %q" "${mode_args[@]}"
    printf " --override %q" "input.split_dir=$SPLIT_DIR"
    printf " --override %q" "runtime.device=cuda"
    printf " --override %q" "runtime.num_workers=$NUM_WORKERS"
    printf " --override %q" "runtime.pin_memory=true"
    printf " --override %q" "runtime.cache_windows=true"
    printf " --override %q" "runtime.mmap_arrays=true"
    printf " --override %q" "optimization.batch_size=$BATCH_SIZE"
    printf " --override %q" "optimization.epochs=$EPOCHS"
    printf " --override %q" "optimization.early_stop_patience=$PATIENCE"
    printf " --override %q" "optimization.allow_subject_overlap_validation=$ALLOW_SUBJECT_OVERLAP"
    printf " --override %q" "output.runs_dir=$RUNS_DIR"
    echo " 2>&1 | tee -a '$task_log'"
    echo "echo '[task-done] session=$session'"
  } > "$task_script"
  chmod +x "$task_script"
  tmux new-session -d -s "$session" "bash '$task_script'"
  echo "started $session on GPU $gpu log=$task_log"
done

cat <<EOF

Launched ${#tasks[@]} jobs. Useful commands:
  tmux ls
  tmux attach -t ${SESSION_PREFIX}_${SPLIT_KIND}_final0_gpu${GPUS[0]}
  tail -f $LOG_DIR/*.log
  watch -n 5 nvidia-smi

Local RTX 4070 Laptop timing was about 156 sec/epoch at batch 16.
On 8x RTX 4090 with batch $BATCH_SIZE, expect roughly 1-3 hours wall time for
the six primary jobs, depending on CPU/disk throughput and early stopping.
Re-estimate from the first '[epoch 001] sec=...' line in each log.
EOF
