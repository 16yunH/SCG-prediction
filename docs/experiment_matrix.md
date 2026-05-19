# Experiment Matrix

This table tracks completed and planned experiments for SCG-based SBP/DBP prediction. Final conclusions should use measured-BP test samples only; interpolated labels are training-only weak labels.

| ID | Protocol | Data | Method | Status | Test SBP MAE | Test DBP MAE | Test Mean MAE | Notes |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| E00 | Subject-independent | v2 measured windows | Mean baseline | Done | 11.975 | 7.836 | 9.905 | Current best SI reference. |
| E01 | Subject-independent | v2 measured windows | Ridge features | Done | 12.801 | 8.631 | 10.716 | Worse than mean baseline. |
| E02 | Subject-independent | v2 measured windows | CNN-only | Done | 13.436 | 8.699 | 11.068 | Deep model overfits / weak cross-subject transfer. |
| E03 | Calibrated | v2 measured windows | CNN-only | Done | 6.253 | 5.659 | 5.956 | Best basic deep ablation. |
| E04 | Calibrated | v2 measured windows | CNN ensemble | Done | 4.864 | 4.348 | 4.606 | Previous best deep result. |
| E05 | Calibrated | v2 measured windows | Random forest residual | Done | 2.894 | 3.674 | 3.284 | Current best calibrated reference. |
| E06 | Both | v3 measured + interpolated + unlabeled index | Data generation | Done |  |  |  | 13,713 supervised windows, 17,019 unlabeled windows, supervised coverage about 19.6%. |
| E07 | Calibrated | v3 measured + interpolated | TCN, final split, 2 epochs | Smoke done | 8.815 | 6.280 | 7.547 | Local RTX 4070 Laptop 8GB run; validates GPU/data path, not convergence. |
| E10 | Calibrated | v3 measured + interpolated | Subject mean / residual ridge / residual RF | Pending |  |  |  | Re-check whether v3 windows improve the strongest calibrated baselines. |
| E11 | Calibrated | v3 measured + interpolated | TCN full training | Pending |  |  |  | Primary local GPU deep run; start with batch 16, 80-200 epochs, early stop 20-30. |
| E12 | Calibrated | v3 measured only | TCN full training | Pending |  |  |  | Ablates whether interpolation helps or hurts. |
| E13 | Calibrated | v3 measured + interpolated | TCN pseudo-label weight sweep | Pending |  |  |  | Compare weak-label weights 0.1, 0.2, 0.5. |
| E14 | Subject-independent | v3 measured + interpolated | Mean / ridge / residual RF | Pending |  |  |  | Must beat E00 to claim cross-subject improvement. |
| E15 | Subject-independent | v3 measured + interpolated | TCN full training | Pending |  |  |  | Hardest protocol; expect lower priority if it remains below mean baseline. |
| E16 | Both | v3 unlabeled windows | Self-supervised pretraining + fine-tune | Not implemented |  |  |  | Data index exists; pretraining task still needs implementation. |

## Local GPU Defaults

Use `myenv` for training. For the 8GB RTX 4070 Laptop GPU, start conservatively:

```powershell
conda activate myenv
python -m src train --model tcn --config configs/train_v3.yaml `
  --override input.split_dir=./artifacts/processed/v3/calibrated_splits `
  --override optimization.allow_subject_overlap_validation=true `
  --override runtime.device=cuda `
  --override runtime.num_workers=0 `
  --override runtime.pin_memory=false `
  --override optimization.batch_size=16
```

Increase `batch_size` only after confirming memory headroom. Use `--mode final` for quick iteration and full `all` mode for publishable CV + test summaries.
