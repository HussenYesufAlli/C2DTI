# C2DTI: High-Level to Deep Code Explanation

This document explains the C2DTI pipeline from the big picture down to class/function-level behavior.

## 1. High-Level Overview

C2DTI execution has three layers:

1. Entry and orchestration
- CLI entry script dispatches run mode.
- Runner loads config, validates, executes pipeline, writes outputs.

2. Core ML pipeline
- Load dataset.
- Split into train/test masks.
- Build predictor from config.
- Fit/predict on training mask.
- Evaluate on test mask.

3. Optional causal layer
- Perturb interactions.
- Compute selected causal mode (reliability, cross_view, mas, irm_cf, unified).
- Save causal metrics and final causal score.

## 2. End-to-End Call Flow

Main command:

```bash
python scripts/run.py --config <config.yaml> --run-once
```

Runtime order:

1. `scripts/run.py`
- Parses flags (`--dry-run`, `--run-once`, `--check-data`).
- Calls `dry_run` or `run_once` in `src/c2dti/runner.py`.

2. `src/c2dti/runner.py::run_once`
- Reads YAML config.
- Calls `validate_config`.
- Creates run directory and writes config snapshot.
- Loads dataset via `load_dti_dataset`.
- Creates predictor with `create_predictor`.
- Builds train/test masks via `split_dataset`.
- Calls predictor `fit_predict`.
- Evaluates metrics via `evaluate_predictions`.
- Optionally computes causal metrics.
- Writes `summary.json`, `predictions.csv`, and registry row.

## 3. Config Validation Layer

File: `src/c2dti/config_validation.py`

Key functions:

1. `_validate_dataset_config`
- Checks dataset name/path and placeholder policy type.

2. `_validate_model_config`
- Validates model name and per-model hyperparameters.

3. `_validate_perturbation_config`
- Validates perturbation strength and seed.

4. `_validate_split_config`
- Validates split strategy and ratio.

5. `validate_config`
- Root-level coordinator.
- Calls `validate_causal_config` from `src/c2dti/causal_objective.py`.

## 4. Dataset Layer

File: `src/c2dti/dataset_loader.py`

Main classes/functions:

1. `DTIDataset` (dataclass)
- Unified container with `drugs`, `targets`, `interactions`, `metadata`.

2. `DTIDatasetLoader` (abstract base)
- Common loader interface (`load`).

3. `BindingDBLoader`
- Reads BindingDB CSV.
- Normalizes possible column variants.
- Builds interaction matrix from Drug/Target identifiers (or content when present).

4. `_MatrixCSVLoader`
- Shared implementation for DAVIS and KIBA flat CSV format:
  - `Drug_ID, Drug, Target_ID, Target, Y`
- Reconstructs ordered entities by IDs.
- Fills dense matrix.

5. `DAVISLoader`, `KIBALoader`
- Thin wrappers over `_MatrixCSVLoader`.

6. `create_dataset_loader`
- Factory by dataset name.

7. `load_dti_dataset`
- Convenience one-call loader.

## 5. Split Layer

File: `src/c2dti/splitter.py`

Core functions:

1. `_random_split`
- Random known-pair split.

2. `_cold_drug_split`
- Holds out entire drugs for test set.

3. `_cold_target_split`
- Holds out entire targets for test set.

4. `split_dataset`
- Public strategy dispatcher and validator.

Output of split stage:
- `train_mask`: boolean matrix of train entries.
- `test_mask`: boolean matrix of held-out entries.

## 6. Predictor Layer

File: `src/c2dti/dti_model.py`

### Shared helper functions

1. `_row_softmax`
2. `_row_normalize`
3. `_cosine_affinity`
4. `_topk_adjacency`
5. `_prepare_training_view`

These support normalization, graph construction, and train-view masking.

### Predictor interface

6. `DTIPredictor`
- Abstract `fit_predict(dataset, train_mask)` contract.

### Predictor implementations

7. `SimpleMatrixDTIPredictor`
- Non-trainable baseline.
- Combines row/column/global statistics plus hashed feature prior.

8. `DualFrozenBackbonePredictor`
- Phase-1 sequence-view predictor.
- Loads frozen embeddings (or hash fallback).
- Uses cosine prior + statistics prior + calibration on train entries.

9. `MatrixFactorizationDTIPredictor`
- Trainable low-rank factorization (`P @ Q.T`).
- Gradient descent on train entries only.
- Exposes `train_loss_history`.
- Can save embeddings checkpoint (`save_checkpoint`).

10. `MixHopPropagationDTIPredictor`
- Graph propagation baseline.
- Builds drug and target similarity graphs.
- Multi-hop propagation with weighted blending.

11. `InteractionCrossAttentionDTIPredictor`
- Two-branch fusion:
  - Branch A: matrix factorization score.
  - Branch B: cross-attention prior from hashed features.
- Top-k sparsifies attention and calibrates branch fusion on train entries.

12. `create_predictor`
- Factory from model config/name.
- Supported models:
  - `simple_baseline`
  - `dual_frozen_backbone`
  - `matrix_factorization`
  - `mixhop_propagation`
  - `interaction_cross_attention`

## 7. Backbone and Feature Utilities

File: `src/c2dti/backbones.py`

1. `load_frozen_entity_embeddings`
- Loads embeddings from NPZ (ordered or keyed formats).
- Falls back to deterministic hash features when NPZ is missing/incompatible.

2. `MASHead`
- Numpy masked autoencoder head.
- Masks fixed dimensions, fits least-squares decoder, computes reconstruction MSE.

File: `src/c2dti/data_utils.py`

3. `build_string_feature_matrix`
- Deterministic hashed character-based feature vectors.

4. `summarize_matrix`
- Mean/std/min/max summary for prediction matrices.

## 8. Evaluation Layer

File: `src/c2dti/evaluation.py`

Metrics:

1. `compute_mse`
2. `compute_rmse`
3. `compute_pearson`
4. `compute_spearman`
5. `compute_ci`
6. `evaluate_predictions` (wrapper)

In split mode, evaluation is done on test-mask entries only.

## 9. Causal Layer

File: `src/c2dti/causal_objective.py`

1. `validate_causal_config`
- Validates causal block and mode.

2. `compute_causal_reliability_score`
- Baseline perturbation stability score.

3. `compute_cross_view_causal_metrics` (Pillar 2)
- Cross-view MSE terms and bounded score.

4. `compute_mas_losses` (Pillar 3)
- Drug/protein MAS reconstruction losses.

5. `compute_irm_cf_losses` (Pillar 4 wrapper)
- Calls IRM and counterfactual helpers and returns combined outputs.

File: `src/c2dti/irm_loss.py`

6. `compute_irm_penalty`
- Environment-wise MSE variance across drug partitions.

7. `compute_counterfactual_loss`
- Scores swapped-target hard negatives.

File: `src/c2dti/unified_scorer.py`

8. `UnifiedC2DTIScorer`
- Combines all active causal pillars with lambda weights.
- Computes `l_total` and `unified_causal_score`.

File: `src/c2dti/perturbation.py`

9. `perturb_dataset_interactions`
- Masks interaction matrix entries for intervention-style stress testing.

## 10. Output Layer

File: `src/c2dti/output_io.py`

1. `make_run_dir`
2. `write_summary`
3. `write_config_snapshot`
4. `write_prediction_matrix`
5. `append_registry`

Artifacts per run:

- `summary.json`
- `config_snapshot.yaml`
- `predictions.csv`
- optional checkpoint
- row in `outputs/results_registry.csv`

## 11. What to Read First (Recommended Learning Path)

For a quick understanding path:

1. `scripts/run.py`
2. `src/c2dti/runner.py`
3. `src/c2dti/dataset_loader.py`
4. `src/c2dti/dti_model.py`
5. `src/c2dti/splitter.py`
6. `src/c2dti/evaluation.py`
7. `src/c2dti/causal_objective.py`
8. `src/c2dti/unified_scorer.py`
9. `src/c2dti/output_io.py`

## 12. Practical Mental Model

Think of the pipeline as two stacked loops:

1. Prediction loop (core DTI):
- Data -> Split -> Predictor -> Metrics

2. Causal loop (robustness/invariance):
- Perturbation + auxiliary views/embeddings -> Causal losses -> Causal score

Both loops are orchestrated in one run contract by `run_once`.
