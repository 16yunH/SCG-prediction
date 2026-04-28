#!/usr/bin/env bash
set -euo pipefail

# Bootstrap conda env on server.
# Example:
# bash scripts/remote/bootstrap_server.sh --env-name lsw-bp --project-dir /home/jiajie/yhong/lsw/project

ENV_NAME="lsw-bp"
PROJECT_DIR="/home/jiajie/yhong/lsw/project"
PYTHON_VERSION="3.10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2;;
    --project-dir) PROJECT_DIR="$2"; shift 2;;
    --python-version) PYTHON_VERSION="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda activate "$ENV_NAME"
cd "$PROJECT_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Environment ready: $ENV_NAME"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
