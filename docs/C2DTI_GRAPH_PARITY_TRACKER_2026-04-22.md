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

## Aggregated Results (Run-once Parity Matrix, Dataset-backed Data)

Evaluation keys used below are from `summary.json -> evaluation_metrics` (`rmse`, `ci`) for the latest 18 parity runs on 2026-04-22.

### Mean by dataset

| Dataset | Branch | RMSE mean | CI mean |
| --- | --- | --- | --- |
| DAVIS | mixhop_baseline | 8570.581126 | 0.500000 |
| DAVIS | interaction_cross_attn | 8570.581126 | 0.500000 |
| KIBA | mixhop_baseline | 5.437395 | 0.500000 |
| KIBA | interaction_cross_attn | 5.437395 | 0.500000 |
| BindingDB | mixhop_baseline | 0.058411 | 0.807633 |
| BindingDB | interaction_cross_attn | 0.058439 | 0.541632 |

### Delta (interaction_cross_attn - mixhop_baseline)

| Dataset | Delta RMSE (lower better) | Delta CI (higher better) |
| --- | --- | --- |
| DAVIS | +0.000000 | +0.000000 |
| KIBA | +0.000000 | +0.000000 |
| BindingDB | +0.000028 | -0.266001 |

## Interpretation and limitation

- The dataset-backed parity matrix completed 18/18 runs.
- During execution, BindingDB initially failed at run 13/18 due to mixed-type ID parsing; this was fixed by normalizing `Drug_ID` and `Target_ID` to strings in the BindingDB loader.
- DAVIS and KIBA are showing flat branch parity in this scaffold baseline setup; BindingDB currently favors mixhop on CI.
- Regression metrics are now from full-scale splits, but scientific promotion still requires protocol-level alignment with MINDG_CLASSA causal evaluation framing.

## Decision Use

Use this tracker for parity verification only.

- If C2DTI parity matches MINDG_CLASSA trend, keep migration plan unchanged.
- If C2DTI trend diverges significantly, open a parity-debug task before promotion.
