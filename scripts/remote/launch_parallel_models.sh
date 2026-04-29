#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper. Prefer launch_ablation_matrix.sh for queue-based GPU use.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$DIR/launch_ablation_matrix.sh" "$@"
