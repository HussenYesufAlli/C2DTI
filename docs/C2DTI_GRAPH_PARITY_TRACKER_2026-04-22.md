# C2DTI Graph Parity Tracker (2026-04-22)

This tracker is the C2DTI-side parity step after MINDG_CLASSA gate completion.

Matrix definition:

- Branches: mixhop_baseline, interaction_cross_attn
- Datasets: DAVIS, KIBA, BindingDB
- Seeds: 10, 34, 42
- Total runs: 18

## Command Generator

Print commands only (recommended first):

```bash
cd /home/hussen/MINDG/C2DTI
python scripts/run_graph_parity_matrix.py --mode dry-run
```

Execute dry-run validation for all 18 configs:

```bash
cd /home/hussen/MINDG/C2DTI
python scripts/run_graph_parity_matrix.py --mode dry-run --execute
```

Execute full run-once matrix:

```bash
cd /home/hussen/MINDG/C2DTI
python scripts/run_graph_parity_matrix.py --mode run-once --execute
```

Generated configs are written to:

- /home/hussen/MINDG/C2DTI/configs/generated_graph_parity/

## Run Matrix

| ID | Dataset | Branch | Seed | Status | Run Name | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | DAVIS | mixhop_baseline | 10 | Completed | C2DTI_DAVIS_MIXHOP_PARITY_S10 |  |
| 02 | DAVIS | mixhop_baseline | 34 | Completed | C2DTI_DAVIS_MIXHOP_PARITY_S34 |  |
| 03 | DAVIS | mixhop_baseline | 42 | Completed | C2DTI_DAVIS_MIXHOP_PARITY_S42 |  |
| 04 | DAVIS | interaction_cross_attn | 10 | Completed | C2DTI_DAVIS_INTERACTION_CROSS_ATTN_PARITY_S10 |  |
| 05 | DAVIS | interaction_cross_attn | 34 | Completed | C2DTI_DAVIS_INTERACTION_CROSS_ATTN_PARITY_S34 |  |
| 06 | DAVIS | interaction_cross_attn | 42 | Completed | C2DTI_DAVIS_INTERACTION_CROSS_ATTN_PARITY_S42 |  |
| 07 | KIBA | mixhop_baseline | 10 | Completed | C2DTI_KIBA_MIXHOP_PARITY_S10 |  |
| 08 | KIBA | mixhop_baseline | 34 | Completed | C2DTI_KIBA_MIXHOP_PARITY_S34 |  |
| 09 | KIBA | mixhop_baseline | 42 | Completed | C2DTI_KIBA_MIXHOP_PARITY_S42 |  |
| 10 | KIBA | interaction_cross_attn | 10 | Completed | C2DTI_KIBA_INTERACTION_CROSS_ATTN_PARITY_S10 |  |
| 11 | KIBA | interaction_cross_attn | 34 | Completed | C2DTI_KIBA_INTERACTION_CROSS_ATTN_PARITY_S34 |  |
| 12 | KIBA | interaction_cross_attn | 42 | Completed | C2DTI_KIBA_INTERACTION_CROSS_ATTN_PARITY_S42 |  |
| 13 | BindingDB | mixhop_baseline | 10 | Completed | C2DTI_BINDINGDB_MIXHOP_PARITY_S10 |  |
| 14 | BindingDB | mixhop_baseline | 34 | Completed | C2DTI_BINDINGDB_MIXHOP_PARITY_S34 |  |
| 15 | BindingDB | mixhop_baseline | 42 | Completed | C2DTI_BINDINGDB_MIXHOP_PARITY_S42 |  |
| 16 | BindingDB | interaction_cross_attn | 10 | Completed | C2DTI_BINDINGDB_INTERACTION_CROSS_ATTN_PARITY_S10 |  |
| 17 | BindingDB | interaction_cross_attn | 34 | Completed | C2DTI_BINDINGDB_INTERACTION_CROSS_ATTN_PARITY_S34 |  |
| 18 | BindingDB | interaction_cross_attn | 42 | Completed | C2DTI_BINDINGDB_INTERACTION_CROSS_ATTN_PARITY_S42 |  |

## Aggregated Results (Run-once Parity Matrix)

Evaluation keys available in current C2DTI summaries are regression-oriented (`mse`, `rmse`, `ci`).

### Mean +/- std by dataset

| Dataset | Branch | RMSE mean +/- std | CI mean +/- std |
| --- | --- | --- | --- |
| DAVIS | mixhop_baseline | 0.383333 +/- 0.000000 | 0.000000 +/- 0.000000 |
| DAVIS | interaction_cross_attn | 0.470931 +/- 0.088944 | 0.000000 +/- 0.000000 |
| KIBA | mixhop_baseline | 0.383333 +/- 0.000000 | 0.000000 +/- 0.000000 |
| KIBA | interaction_cross_attn | 0.470931 +/- 0.088944 | 0.000000 +/- 0.000000 |
| BindingDB | mixhop_baseline | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |
| BindingDB | interaction_cross_attn | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 |

### Delta (interaction_cross_attn - mixhop_baseline)

| Dataset | Delta RMSE (lower better) | Delta CI (higher better) |
| --- | --- | --- |
| DAVIS | +0.087598 | +0.000000 |
| KIBA | +0.087598 | +0.000000 |
| BindingDB | +0.000000 | +0.000000 |

## Interpretation and limitation

- The C2DTI parity matrix executed successfully end-to-end (18/18).
- Current parity numbers are not publication-grade due to very small demo-style data regimes (for example, `n_test=1` in sample summaries).
- Use this step as pipeline/contract parity verification, not final scientific model ranking.
- Final migration decisions should continue to rely on MINDG_CLASSA full-data gate until C2DTI is wired to real benchmark-scale data.

## Decision Use

Use this tracker for parity verification only.

- If C2DTI parity matches MINDG_CLASSA trend, keep migration plan unchanged.
- If C2DTI trend diverges significantly, open a parity-debug task before promotion.
