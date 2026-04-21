# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Repository scaffold for C2DTI.
- Professional git workflow documentation.
- Minimal `unittest` contract checks for config validation and run artifact outputs.
- GitHub Actions workflow to run unit tests on push and pull requests.
- Standard pull request template for supervisor-friendly change reporting.
- Release Go/No-Go checklist for promoting `dev` to `main`.
- Structured GitHub issue templates for bug reports and feature requests.
- Minimal CODEOWNERS policy for automatic review assignment on key paths.
- README status badge section for CI and workflow visibility.
- Minimal Makefile targets (`test`, `smoke`) for standardized local checks.
- Two-minute supervisor demo guide with scripted verification flow.
- Runner failure-path tests for missing and invalid configuration handling.
- CI now uploads unit-test log artifacts for review evidence on each run.
- Demo script now includes quick troubleshooting for common live-run issues.
- Release checklist now includes an explicit Definition of Done section.
- Release checklist now includes owner and reviewer sign-off fields.
- **Non-breaking causal objective hook**: optional `causal` config section with `enabled` and `weight` keys; when enabled, computes placeholder causal score and includes it in run summary.
- Causal objective module with validation, score computation, and comprehensive unit tests (8 new tests, all passing).
- Example config `configs/causal_enabled.yaml` showing causal feature usage.
- Optional real DTI pipeline path with dataset loaders for BindingDB, DAVIS, and KIBA.
- Simple baseline DTI predictor, prediction CSV artifact output, and perturbation-based causal reliability scoring.
- Example real-pipeline config `configs/davis_real_pipeline.yaml` and integration coverage for end-to-end dataset-to-output execution.
- Strict dataset mode with `dataset.allow_placeholder: false` to fail fast when real files are missing.
- Strict real-run config templates for DAVIS, BindingDB, and KIBA (`*_real_pipeline_strict.yaml`).
- Dedicated dataset precheck command via `python scripts/run.py --config <config> --check-data`.
- Dataset precheck now writes reusable JSON reports under `outputs/checks/` with file status and dataset summary.
- Dataset precheck reports now include dataset-specific schema details to guide data preparation.
- BindingDB dataset precheck now validates CSV headers and records available, resolved, and missing columns in the JSON report.
- DAVIS/KIBA dataset precheck now validates sequence file line counts and `Y.txt` shape consistency, and records results in the JSON report.
- Added `scripts/check_all_data.py` and `make check-data-all` to run strict prechecks for DAVIS/BindingDB/KIBA with one summary.
- `check-data-all` now reads generated reports and prints a next-actions checklist (missing files/content fixes) per failed config.
- Strict precheck now rejects BindingDB CSVs with header-only content and DAVIS/KIBA datasets with empty sequence files.
- Added `scripts/scaffold_data_layout.py` and `make scaffold-data-layout` to create required dataset file paths quickly.
- Added `scripts/fill_demo_data.py` and `make fill-demo-data` to populate scaffolded files with minimal synthetic data for strict-check validation.
- Added `scripts/run_all_once.py` and `make run-once-all` to execute strict configs end-to-end with one summary.
- Added `make real-all` to run strict prechecks and strict run-once execution in a single command.
- Added `make gate-all` to run verify plus full strict real pipeline checks in one quality gate.
- Added `scripts/gate_all.py`; `make gate-all` now emits `outputs/gates/gate_all_<timestamp>.json` with step-level gate evidence.

## [0.1.0] - 2026-04-21

### Added
- Initial public repository setup (`main`, `dev`).
- Baseline project structure and configuration folders.
