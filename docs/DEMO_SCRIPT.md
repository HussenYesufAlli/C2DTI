# C2DTI 2-Minute Demo Script

Use this script to present a clean, repeatable project health check for supervisor review.

## Goal

Show that the repository has:

1. Reproducible tests
2. Reproducible smoke run contract
3. Structured artifacts and governance files

## Demo Flow (About 2 Minutes)

### 1) Confirm branch and latest changes

```bash
git status --short
git log --oneline -5
```

Expected:

- Working tree clean or only intended changes
- Recent commits align with changelog and docs

### 2) Run unit tests

```bash
make test
```

Say:

- This validates the minimal contract checks for config validation and run outputs.

### 3) Run smoke contract

```bash
make smoke
```

Say:

- This verifies dry-run and run-once paths.
- Run-once should emit `summary.json`, `config_snapshot.yaml`, and update `results_registry.csv`.

### 4) Point to key governance files

Show quickly:

- `README.md` for project goals and standard commands
- `docs/REPO_WORKFLOW.md` for branch and promotion policy
- `docs/RELEASE_CHECKLIST.md` for Go/No-Go release gating
- `.github/PULL_REQUEST_TEMPLATE.md` for review quality

### 5) Close with status

State:

- CI workflow is configured to run tests on push/PR.
- Current baseline is stable, and next method steps can be added incrementally.

## Optional Backup Commands

If `make` is unavailable:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run.py --config configs/minimal.yaml --dry-run
python scripts/run.py --config configs/minimal.yaml --run-once
```

## Quick Troubleshooting

### `make: command not found`

Use backup commands directly:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run.py --config configs/minimal.yaml --dry-run
python scripts/run.py --config configs/minimal.yaml --run-once
```

### `ModuleNotFoundError` while running scripts

Run commands from the repository root:

```bash
cd /home/hussen/MINDG/C2DTI
```

Then re-run `make test` or `make smoke`.

### Config validation failure in smoke run

Confirm required keys exist in `configs/minimal.yaml`:

- `name`
- `protocol`
- `output.base_dir`

### Need CI evidence for supervisor review

Open the latest GitHub Actions run and download artifact `unittest-log`.
