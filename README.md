# C2DTI

Cross-view Causal DTI (C2DTI) is a research repository for drug-target interaction prediction with multimodal causal consistency. The core idea is a cross-view causal agreement objective that enforces consistency between sequence and graph evidence under perturbation.

## Status

[![Tests](https://github.com/HussenYesufAlli/C2DTI/actions/workflows/tests.yml/badge.svg)](https://github.com/HussenYesufAlli/C2DTI/actions/workflows/tests.yml)
[![Branch](https://img.shields.io/badge/branch-dev-blue)](https://github.com/HussenYesufAlli/C2DTI/tree/dev)
[![PR%20Policy](https://img.shields.io/badge/PR-feature%2F*%20to%20dev-informational)](https://github.com/HussenYesufAlli/C2DTI/blob/main/docs/REPO_WORKFLOW.md)

## Repository Goals

1. Reproduce the baseline pipeline under a fixed protocol.
2. Implement controlled causal extensions without breaking legacy behavior.
3. Track experiments and manuscript artifacts with reproducible git workflow.

## Project Structure

- `src/`: implementation code
- `configs/`: experiment configs
- `scripts/`: execution helpers (to be added as implementation proceeds)
- `docs/`: method and experiment documentation
- `tests/`: test scaffolding
- `data/`: local data placeholders only
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

Commit in small units, then open a PR from `feature/*` to `dev`.

## Common Commands

```bash
make test
make smoke
python scripts/run.py --config configs/davis_real_pipeline_strict.yaml --check-data
make check-data-all
make scaffold-data-layout
make fill-demo-data
make run-once-all
```

Use `--check-data` before a strict real run to verify required files and
matrix shape without starting the prediction pipeline.

Each dataset precheck also writes a reusable JSON report under
`outputs/checks/<config_name>_data_check.json`.

That report now includes dataset-specific schema details, such as expected
BindingDB columns and DAVIS/KIBA required file names.

For BindingDB, the precheck also validates the CSV header and records which
required columns were found, resolved via aliases, or still missing.
It also requires at least one non-empty data row in the CSV.

For DAVIS/KIBA, the precheck now validates that:
- line counts in `drug_smiles.txt` and `target_sequences.txt` are readable
- `Y.txt` can be parsed
- parsed `Y.txt` shape matches `[num_drugs, num_targets]`
- non-empty drug/target entries are present (empty files are rejected)

Use `make check-data-all` to run strict prechecks for DAVIS, BindingDB,
and KIBA in one shot with a compact pass/fail summary.

When checks fail, `make check-data-all` now prints a next-actions checklist
derived from JSON reports (for example, exact missing file paths to create).

Use `make scaffold-data-layout` to create the required `data/` file structure
quickly before filling real dataset contents.

Use `make fill-demo-data` to populate scaffolded files with minimal synthetic
content that satisfies strict checks for quick pipeline validation.

Use `make run-once-all` to execute all strict configs once and get a compact
pass/fail summary for end-to-end run contracts.

For a supervisor walkthrough, use [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## Current Focus

Initial scaffold is complete. Next implementation phase is controlled integration and evaluation of the cross-view causal agreement objective.
