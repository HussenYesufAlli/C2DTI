# C2DTI Run Replay (Debugger Style)

This document replays two real completed runs as if we are stepping through a debugger.

Compared runs:

1. `C2DTI_EVAL_DAVIS_RANDOM_FULL_S10`
2. `C2DTI_EVAL_DAVIS_RANDOM_FULL_S34`

Source files:

1. `outputs/runs/C2DTI_EVAL_DAVIS_RANDOM_FULL_S10-20260423-003130/summary.json`
2. `outputs/runs/C2DTI_EVAL_DAVIS_RANDOM_FULL_S34-20260423-003133/summary.json`

## 1. Command and Entry

Command form:

```bash
python scripts/run.py --config <generated_eval_config.yaml> --run-once
```

Call chain:

1. `scripts/run.py` parses flags.
2. Calls `run_once` in `src/c2dti/runner.py`.

## 2. Config Snapshot (What was requested)

Common across both runs:

1. Dataset: DAVIS
2. Model: `interaction_cross_attention`
3. Split: `random`, test ratio `0.2`
4. Causal mode: `unified`
5. Lambdas: xview=1.0, mas=1.0, irm=1.0, cf=1.0

Differences:

1. Seed 10 run uses seed 10 in split/model/perturbation/causal sub-configs.
2. Seed 34 run uses seed 34.

## 3. Data Loading Stage

Executed by:

1. `load_dti_dataset` in `src/c2dti/dataset_loader.py`
2. For DAVIS: `DAVISLoader` -> `_MatrixCSVLoader`

Observed output in both runs:

1. `num_drugs = 68`
2. `num_targets = 379`
3. `dataset_placeholder = false` (real data, not fallback)

## 4. Split Stage

Executed by:

1. `split_dataset` in `src/c2dti/splitter.py`

Results:

1. Run S10: `n_train=20618`, `n_test=5154`
2. Run S34: `n_train=20618`, `n_test=5154`

Interpretation:

1. Same split ratio and same dataset size produce same counts.
2. Pair assignments differ due to different seed.

## 5. Prediction Stage

Main predictor constructed by `create_predictor` in `src/c2dti/dti_model.py`:

1. `InteractionCrossAttentionDTIPredictor`

Recorded prediction statistics:

1. S10: mean=1.0, std=0.0, min=1.0, max=1.0
2. S34: mean=1.0, std=0.0, min=1.0, max=1.0

Interpretation:

1. Both runs collapsed to constant predictions (`all ones`).
2. This directly explains weak ranking/correlation metrics.

## 6. Evaluation Stage

Executed by `evaluate_predictions` in `src/c2dti/evaluation.py`.

### Test metrics

S10:

1. MSE: 73535028.26719029
2. RMSE: 8575.256746429828
3. Pearson: 0.0
4. Spearman: 0.0
5. CI: 0.5
6. n_evaluated: 5154

S34:

1. MSE: 73339428.09067392
2. RMSE: 8563.8442355448
3. Pearson: 0.0
4. Spearman: 0.0
5. CI: 0.5
6. n_evaluated: 5154

### Train metrics

S10:

1. MSE: 72903189.53091456
2. RMSE: 8538.33646156642
3. Pearson: 0.0
4. Spearman: 0.0
5. CI: 0.5
6. n_evaluated: 20618

S34:

1. MSE: 72952084.8316113
2. RMSE: 8541.199261907623
3. Pearson: 0.0
4. Spearman: 0.0
5. CI: 0.5
6. n_evaluated: 20618

Interpretation:

1. `CI=0.5` indicates random ranking behavior.
2. `Pearson=0`, `Spearman=0` match constant-output collapse.

## 7. Causal Stage (Unified Mode)

The runner uses `UnifiedC2DTIScorer` in `src/c2dti/unified_scorer.py`.

### 7.1 Pillar 2 (Cross-view)

S10:

1. mse_seq_graph: 0.0051087671890854836
2. mse_seqpert_graph: 0.005703864619135857
3. mse_seq_graphpert: 0.0051087671890854836
4. l_xview: 0.015921398997306824

S34:

1. mse_seq_graph: 0.005665062926709652
2. mse_seqpert_graph: 0.005707834381610155
3. mse_seq_graphpert: 0.005665062926709652
4. l_xview: 0.01703796023502946

### 7.2 Pillar 3 (MAS)

S10:

1. mas_drug_loss: 5.250084213513833e-05
2. mas_prot_loss: 3.336272268071133e-05
3. l_mas: 8.586356481584966e-05

S34:

1. mas_drug_loss: 9.892035467571245e-06
2. mas_prot_loss: 6.110242528025029e-05
3. l_mas: 7.099446074782153e-05

### 7.3 Pillar 4 (IRM + Counterfactual)

S10:

1. l_irm: 22339376338068.934
2. l_cf: 1.0
3. n_cf_sampled: 1000
4. env_mses: [68744281.91581725, 73214770.84849808, 80672176.3532263, 69486960.76876836]

S34:

1. l_irm: 10069208883404.242
2. l_cf: 1.0
3. n_cf_sampled: 1000
4. env_mses: [78310624.56897607, 71374609.31000842, 72456195.33341219, 69976760.6739133]

### 7.4 Unified aggregation

S10:

1. l_irm_normalised: 1.0
2. l_total: 2.0160072625621224
3. unified_causal_score: 0.3315641883270838

S34:

1. l_irm_normalised: 1.0
2. l_total: 2.017108954695777
3. unified_causal_score: 0.3314431182352951

Interpretation:

1. Unified causal score is stable across seeds here (~0.3315).
2. Main performance bottleneck remains the prediction collapse, not causal computation instability.

## 8. Output Writing Stage

Handled by `src/c2dti/output_io.py`:

1. `predictions.csv` written.
2. `summary.json` written.
3. Config snapshot saved.
4. Registry row appended to `outputs/results_registry.csv`.

## 9. Side-by-Side Delta Summary

| Item | S10 | S34 | Observation |
|---|---:|---:|---|
| Test CI | 0.5 | 0.5 | identical random ranking |
| Test RMSE | 8575.2567 | 8563.8442 | very close |
| Pearson | 0.0 | 0.0 | both collapsed |
| Spearman | 0.0 | 0.0 | both collapsed |
| l_total | 2.0160 | 2.0171 | stable causal aggregate |
| unified_causal_score | 0.33156 | 0.33144 | stable across seeds |

## 10. Key Practical Takeaways

1. Infrastructure is functioning end-to-end (load, split, predict, evaluate, causal, persist).
2. Current predictor configuration is the limiting factor (constant predictions).
3. Before scaling to all 135 runs, improve predictor behavior to avoid wasting compute.
4. Causal layer itself is numerically stable enough to continue once predictor quality improves.
