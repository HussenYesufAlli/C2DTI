# C2DTI

Cross-view Causal DTI (C2DTI) is a research repository for drug-target interaction prediction with multimodal causal consistency. The core idea is a cross-view causal agreement objective that enforces consistency between sequence and graph evidence under perturbation.

## Project Structure

- `src/`: implementation code
- `configs/`: experiment configs
- `scripts/`: execution helpers (to be added as implementation proceeds)
- `docs/`: method and experiment documentation
- `tests/`: test scaffolding
- `datasets/`: local data placeholders only
- `outputs/`: runtime artifacts (ignored)
- `models/`: checkpoints (ignored)

## Branch Strategy

- `main`: stable and reproducible snapshots
- `dev`: integration branch
- `feature/*`: one branch per method/experiment/doc task

## Quick Start (Git Workflow)

```bash
git checkout dev
git pull origin dev
git checkout -b feature/<task-name>
```

Use `--check-data` before a strict dataset run to verify required files and
matrix shape without starting the prediction pipeline.

For MixHop-to-C2DTI migration decisions, use
[docs/MIXHOP_TO_C2DTI_MIGRATION_GATE.md](docs/MIXHOP_TO_C2DTI_MIGRATION_GATE.md).

For C2DTI-side parity execution tracking, use
[docs/C2DTI_GRAPH_PARITY_TRACKER_2026-04-22.md](docs/C2DTI_GRAPH_PARITY_TRACKER_2026-04-22.md).


## Phase 6 Evaluation Matrix

Run the full causal evaluation matrix (regression, 135 runs: 3 datasets × 3 splits × 5 ablations × 3 seeds):

```bash
# Pilot — validate 3 runs before committing to full matrix
python scripts/run_eval_matrix.py --mode run-once --execute --max-runs 3

# Full matrix
python scripts/run_eval_matrix.py --mode run-once --execute
```

Run the binary classification evaluation matrix (27 runs: 3 datasets × 3 splits × 3 seeds):

```bash
# Pilot — validate 3 runs before committing to full matrix
python scripts/run_binary_eval_matrix.py --mode run-once --execute --max-runs 3

# Full matrix
python scripts/run_binary_eval_matrix.py --mode run-once --execute
```

After all runs complete, compile results into CSV tables:

```bash
python scripts/compile_results.py --prefix C2DTI_EVAL_
python scripts/compile_binary_results.py --prefix C2DTI_BINARY_EVAL_
```

Outputs land in `outputs/reports/` and `outputs_binary/reports/` respectively.


## Current Focus

Initial scaffold is complete. Next implementation phase is controlled integration and evaluation of the cross-view causal agreement objective.
