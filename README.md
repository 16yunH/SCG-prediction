# SCG Blood Pressure Prediction

End-to-end pipeline for local development, GitHub sync, structured data preparation, and remote 8-GPU ablation training.

## Project Layout

- `src/scg_bp/`: core data, split, training, evaluation, and report code
- `src/*.py`: public CLI entry modules
- `configs/`: YAML configs
- `scripts/remote/`: server scripts for GPU guarded training and ablation matrix launch
- `tests/`: smoke tests

## Structured Data Pipeline

The v2 data pipeline converts mixed raw files into reusable assets under `artifacts/processed/v2`:

- `raw_manifest.csv`: all scanned files with role and parser status
- `bp_index.csv`: standardized SBP/DBP/HR labels
- `signal_index.csv`: SCG metadata and generated array paths
- `arrays/*.npy`: float32 SCG arrays with 6 selected channels
- `window_index.csv`: trainable windows with labels and `alignment_method`
- `qc_report.json`: subject-level counts, exclusions, and parser failures

```bash
python -m src.prepare_data --config configs/data.yaml
python -m src.make_splits --config configs/split.yaml
```

Server example:

```bash
python -m src.prepare_data --config configs/data.yaml \
  --override paths.data_root=/home/jiajie/yhong/lsw/data \
  --override paths.processed_dir=/home/jiajie/yhong/lsw/artifacts/processed/v2 \
  --override paths.arrays_dir=/home/jiajie/yhong/lsw/artifacts/processed/v2/arrays \
  --override output.raw_manifest=/home/jiajie/yhong/lsw/artifacts/processed/v2/raw_manifest.csv \
  --override output.bp_index=/home/jiajie/yhong/lsw/artifacts/processed/v2/bp_index.csv \
  --override output.signal_index=/home/jiajie/yhong/lsw/artifacts/processed/v2/signal_index.csv \
  --override output.window_index=/home/jiajie/yhong/lsw/artifacts/processed/v2/window_index.csv \
  --override output.sample_index=/home/jiajie/yhong/lsw/artifacts/processed/v2/sample_index.csv \
  --override output.qc_report=/home/jiajie/yhong/lsw/artifacts/processed/v2/qc_report.json \
  --override bp.strict=true

python -m src.make_splits --config configs/split.yaml \
  --override input.sample_index=/home/jiajie/yhong/lsw/artifacts/processed/v2/window_index.csv \
  --override output.split_dir=/home/jiajie/yhong/lsw/artifacts/processed/v2/splits
```

## Training

Single model, all folds plus final holdout:

```bash
python -m src.train --model full --config configs/train.yaml
```

Single fold task, useful for GPU matrix scheduling:

```bash
python -m src.train --model full --config configs/train.yaml --mode cv --fold 1
```

Final holdout-only task:

```bash
python -m src.train --model full --config configs/train.yaml --mode final
```

Training reads `window_index.csv + arrays/*.npy` through split files. It only falls back to raw CSV reads for debugging or legacy indices.

## Remote 8-GPU Ablation Matrix

Use the queue launcher to run model/fold jobs across available GPUs with live tmux output and persistent logs:

```bash
cd ~/yhong/lsw/project
chmod +x scripts/remote/*.sh
bash scripts/remote/launch_ablation_matrix.sh \
  --conda-env yh \
  --project-dir /home/jiajie/yhong/lsw/project \
  --split-dir /home/jiajie/yhong/lsw/artifacts/processed/v2/splits \
  --runs-dir /home/jiajie/yhong/lsw/runs \
  --log-dir /home/jiajie/yhong/lsw/runs/logs \
  --gpu-pool 1,2,3,5,6,7 \
  --folds 5 \
  --workers 4 \
  --batch-size 512 \
  --epochs 30 \
  --include-final true
```

Attach to a worker:

```bash
tmux ls
tmux attach -t ablation_gpu1
```

## Evaluation And Report

```bash
python -m src.evaluate --config configs/eval.yaml \
  --override input.runs_dir=/home/jiajie/yhong/lsw/runs \
  --override output.metrics_summary=/home/jiajie/yhong/lsw/artifacts/metrics/metrics_summary.csv \
  --override output.fold_metrics=/home/jiajie/yhong/lsw/artifacts/metrics/fold_metrics.csv

python -m src.report --config configs/report.yaml \
  --override input.metrics_summary=/home/jiajie/yhong/lsw/artifacts/metrics/metrics_summary.csv \
  --override input.fold_metrics=/home/jiajie/yhong/lsw/artifacts/metrics/fold_metrics.csv \
  --override output.figure_dir=/home/jiajie/yhong/lsw/artifacts/figures \
  --override output.report_md=/home/jiajie/yhong/lsw/artifacts/report.md
```

## Notes

- Legacy `.xls` requires `python-calamine` or `xlrd`; `python-calamine` is preferred.
- Splits are subject-level to avoid leakage.
- Default alignment is `rank_interpolation` because exact BP-SCG timestamps are not consistently available.
- The current dataset is suitable for project-level ablation comparison, not clinical-grade generalization claims.
