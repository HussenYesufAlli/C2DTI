# C2DTI Release Checklist (Dev -> Main)

Use this checklist before promoting `dev` to `main`.
A release is approved only when all Go/No-Go items are `Go`.

## 1. Scope Freeze

- [ ] Go/No-Go: Release scope is explicitly listed in PR description.
- [ ] Go/No-Go: No unrelated feature commits are included.
- [ ] Go/No-Go: Changelog entries for included work are present.

## 2. Validation Gate

- [ ] Go/No-Go: Unit tests pass locally.
- [ ] Go/No-Go: CI workflow is green for latest `dev` commit.
- [ ] Go/No-Go: Minimal run contract still succeeds.

Reference commands:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run.py --config configs/minimal.yaml --dry-run
python scripts/run.py --config configs/minimal.yaml --run-once
```

## 3. Artifact Gate

- [ ] Go/No-Go: Run output contains `summary.json`.
- [ ] Go/No-Go: Run output contains `config_snapshot.yaml`.
- [ ] Go/No-Go: `outputs/results_registry.csv` contains the new run row.

## 4. Review Gate

- [ ] Go/No-Go: PR template sections are completed (purpose, validation, risk).
- [ ] Go/No-Go: At least one reviewer approval is recorded.
- [ ] Go/No-Go: Any reviewer-requested changes are resolved.

## 5. Merge Gate

- [ ] Go/No-Go: Merge target is `main` from `dev`.
- [ ] Go/No-Go: Merge method is documented in PR (merge/squash/rebase).
- [ ] Go/No-Go: Release tag name is prepared (if milestone release).

## 6. Post-Merge Gate

- [ ] Go/No-Go: `main` is pulled and validated once.
- [ ] Go/No-Go: Tag is created/pushed when required.
- [ ] Go/No-Go: Release note link is shared with supervisor.

## 7. Definition Of Done (Release)

A release is considered Done only if all conditions below are true:

- [ ] Every Go/No-Go item in Sections 1-6 is checked as Go.
- [ ] No open blocking issue remains for the release scope.
- [ ] CI status for the merge commit on `main` is green.
- [ ] Supervisor-facing evidence exists (changelog, checklist, and CI/test logs).
- [ ] Rollback path is documented and feasible.

## Decision

- Final Status: [ ] Go  [ ] No-Go
- Release version/tag:
- Date:
- Owner:
- Notes:
