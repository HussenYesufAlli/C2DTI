# C2DTI Repository Workflow

## Daily Workflow

1. Update local dev branch.

```bash
git checkout dev
git pull origin dev
```

2. Create task branch.

```bash
git checkout -b feature/<task>
```

3. Implement changes and validate.
4. Commit small logical units.
5. Push branch and open PR to `dev`.

## Promotion to Main

Merge `dev` to `main` only at stable milestones:

- baseline lock
- causal objective integration
- cross-view objective integration
- submission-ready release

## Tagging Milestones

```bash
git checkout main
git pull origin main
git tag -a v0.1-baseline-locked -m "Baseline and protocol lock"
git push origin --tags
```
