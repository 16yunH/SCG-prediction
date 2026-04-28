# SCG Blood Pressure Prediction

End-to-end pipeline for local development + GitHub sync + remote server training.

## Project Layout

- `src/scg_bp/`: core pipeline and training code
- `src/*.py`: public CLI entry modules
- `configs/`: YAML configs
- `scripts/remote/`: server scripts (`gpu_guard`, `tmux` launcher)
- `scripts/local/`: local helper scripts
- `tests/`: smoke tests

## Public CLI

```bash
python -m src.prepare_data --config configs/data.yaml
python -m src.make_splits --config configs/split.yaml
python -m src.train --model full --config configs/train.yaml
python -m src.train --model cnn_only --config configs/train.yaml
python -m src.train --model lstm_only --config configs/train.yaml
python -m src.train --model mlp_only --config configs/train.yaml
python -m src.evaluate --config configs/eval.yaml
python -m src.report --config configs/report.yaml
```

## Remote training with GPU guard

```bash
bash scripts/remote/run_with_gpu_guard.sh \
  --gpu-util-threshold 30 \
  --gpu-mem-threshold 50 \
  --poll-seconds 60 \
  --session train_full \
  --command "python -m src.train --model full --config configs/train.yaml"
```

## Notes

- BP mixed formats are supported by file signature detection (`xlsx`, misnamed `xlsx`, text CSV, legacy `xls`).
- Legacy `.xls` requires `python-calamine` or `xlrd` on the server.
- Splits are group-based by subject to avoid leakage.
