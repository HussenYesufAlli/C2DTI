# MixHop to C2DTI Migration Gate

This gate defines the required evidence before promoting the new graph-view branch from MINDG_CLASSA into C2DTI.

Goal: keep migration non-breaking, reviewer-safe, and data-driven.

## 1. Migration Scope

- Baseline path must remain available: MixHop default behavior cannot be removed.
- New path must be optional via config flag.
- Initial migration scope is graph-view branch only.
- Causal extras (MAS/IRM/MLM/counterfactual) are phase-2 after baseline parity is verified.

## 2. Required Experiment Matrix

Run the same two graph encoders under matched settings:

- Encoder A: MixHopNetwork (baseline)
- Encoder B: Interaction-aware branch (cross-attention plus intra-layer head)

Datasets:

- DAVIS
- KIBA
- BindingDB_Kd

Seeds:

- 10
- 34
- 42

Minimum runs: 2 encoders x 3 datasets x 3 seeds = 18 runs.

## 3. Locked Protocol

To avoid moving targets during migration decisions:

- Use identical split files between encoder A and B.
- Use identical optimizer, scheduler, epochs, and early-stopping settings.
- Use identical feature backbones and embedding files.
- Select one primary metric per dataset and keep it fixed for all comparisons.

Recommended primary metric:

- AUPRC for classification-centered DTI comparisons.

## 4. Go and No-Go Rules

All rules below are required for a final Go:

- Rule A (safety): Encoder B must be non-inferior to Encoder A on mean AUPRC in at least 2 of 3 datasets.
- Rule B (hard-shift value): Encoder B must show clear gain on at least one harder-shift dataset (typically BindingDB_Kd or cold-start split).
- Rule C (stability): standard deviation across seeds for Encoder B must not be materially worse than Encoder A on the primary metric.
- Rule D (non-breaking): default C2DTI config must still run MixHop path unchanged.

If any rule fails, decision is No-Go for default replacement.
In that case, keep Encoder B as optional ablation path only.

## 5. Evidence Artifacts

Store these artifacts before decision:

- Per-run summaries under outputs with config snapshots.
- Consolidated comparison table (dataset, seed, metric deltas).
- Mean and standard deviation table across seeds.
- Short decision memo with final status: Go or No-Go.

Execution assets:

- Run tracker: /home/hussen/MINDG/MINDG_CLASSA/doc/experiments/GRAPH_GATE_18_RUN_TRACKER_2026-04-22.md
- Command generator: /home/hussen/MINDG/MINDG_CLASSA/scripts/run_graph_gate_matrix.py

## 6. Recommended Decision for Current State

Based on current single-seed evidence:

- BindingDB_Kd: strong positive signal for interaction-aware branch.
- DAVIS: near-neutral.
- KIBA: negative signal.

Current status: HOLD (insufficient for replacement).

Action now:

1. Complete 3-seed matrix.
2. Evaluate rules A-D.
3. If pass: migrate Encoder B to C2DTI as optional first, then consider default.
4. If fail: keep MixHop default and keep Encoder B as ablation branch.

## 7. Decision Record

- Final Status: [ ] Go  [x] No-Go  [x] Hold
- Date: 2026-04-22
- Owner: MINDG_CLASSA -> C2DTI migration gate
- Compared Configs: MINDG_CLASSA graph comparison matrix (run_id=5, 18 runs)
- Primary Metric: test AUPRC
- Notes:
	- Rule A failed (non-inferior on mean AUPRC achieved in 1/3 datasets).
	- Rule B passed (clear gain on BindingDB_Kd).
	- Rule C mixed/failed (higher variance for interaction branch on DAVIS/KIBA).
	- Rule D passed (default MixHop path remains intact).
	- Decision: No-Go for default replacement; Hold as optional branch pending further tuning/splits.

	## 8. C2DTI Parity Execution Note (2026-04-22)

	- C2DTI parity matrix execution status: completed (18/18 run-once jobs).
	- Tracking artifact: /home/hussen/MINDG/C2DTI/docs/C2DTI_GRAPH_PARITY_TRACKER_2026-04-22.md
	- Interpretation: parity run confirms pipeline readiness, but current C2DTI metrics are generated on demo-scale data and should not override the MINDG_CLASSA full-data gate outcome.
