# Contributing to C2DTI

## Branching Rules

1. Do not commit directly to `main`.
2. Use `dev` as integration branch.
3. Create one `feature/*` branch per task.

Example:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/method-cross-view-loss
```

## Commit Message Style

Use conventional prefixes:

- `feat:` new capability
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` restructuring without intended behavior change
- `chore:` setup/tooling updates
- `exp:` experiment config/result updates

Example:

```text
feat: add cross-view agreement loss in causal runner
```

## Pull Request Checklist

Every PR should include:

1. Purpose and motivation.
2. Files/components changed.
3. Validation steps and outcomes.
4. Risk notes and rollback approach.

## Experiment Reporting Rule

A run is reportable only if all of the following are present:

1. Config file.
2. Stored artifacts.
3. Result summary row/record.
4. Short note in `docs/` with interpretation.
