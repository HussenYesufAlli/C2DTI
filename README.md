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

## Current Focus

Initial scaffold is complete. Next implementation phase is controlled integration and evaluation of the cross-view causal agreement objective.
