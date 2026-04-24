# C2DTI Results Analysis

## Scope

This document summarizes the current validation and analysis status produced from the notebook-first Phase 1-7 workflow in this repository.

Current status:
- Module-level implementation checks are completed for backbone, causal objective, IRM/counterfactual, and unified scorer.
- Full evaluation matrix execution is prepared but not yet launched.
- Aggregated benchmark CSV reports are not available yet because no run summaries with prefix `C2DTI_EVAL_` were found at compile time.

## Verified Module Outputs (Notebook)

### 1. Backbone Validation

Backbone checks confirmed expected behavior:
- Frozen embedding loader returns stable shape contracts, including fallback behavior when files are missing or incompatible.
- MAS head reconstruction behavior is numerically stable.
- Sequence view encoder is deterministic and returns normalized fixed-width vectors.

Representative observations:
- Fallback embeddings remained non-zero and shape-consistent.
- MAS demo produced near-zero train-like reconstruction loss and larger shifted-input loss.

### 2. Causal Objective Validation

Causal helper checks completed successfully:
- Config validation accepted valid schemas and rejected invalid values with explicit error messages.
- Cross-view metrics produced finite MSE components and bounded causal score.
- MAS and IRM/CF helper outputs were computed end-to-end from valid tensor inputs.

Representative values from notebook demos:
- Cross-view causal score: about 0.9915
- MAS score: 1.0 in controlled synthetic setup
- IRM+CF score examples: about 0.7568 and 0.6911

### 3. Unified Scorer Validation (Phase 5)

Unified scorer orchestration and ablation mechanics were verified.

Representative values from notebook demos:
- Full-pillars run:
  - l_xview: 0.4839
  - l_mas: ~0
  - l_irm: 0.0055
  - l_cf: 0.5240
  - l_total: 1.0135
  - unified_causal_score: 0.4967
- Example ablation (xview and irm disabled):
  - unified_causal_score: 0.6562
  - delta versus full: +0.1595

Interpretation:
- The unified pipeline responds to lambda ablations as expected.
- Current figures are sanity-check diagnostics from synthetic notebook inputs, not benchmark claims.

## Phase 6 Evaluation Matrix Status

Tooling status:
- `scripts/run_eval_matrix.py` verified and executed in dry-run mode.
- `scripts/compile_results.py` executed successfully.

Current output state:
- Dry-run command sheet generation is functional.
- Compile stage returned: no runs found with prefix `C2DTI_EVAL_`.
- Therefore, no detailed or aggregate benchmark tables are currently available for publication reporting.

## Key Risks Before Final Reporting

1. No real matrix execution yet:
- Statistical performance claims across dataset/split/ablation/seed cannot be made until full matrix runs are executed.

2. Synthetic-check bias:
- Most causal objective and unified scorer numbers documented so far come from controlled notebook arrays.
- These validate code behavior, not final scientific performance.

3. Reporting gap:
- CI/RMSE/Pearson/Spearman tables by dataset and split are pending actual run outputs.

## Recommended Next Analysis Slice

1. Execute matrix runs:
- Launch `scripts/run_eval_matrix.py` with execute mode using controlled run batches.

2. Compile outputs:
- Re-run `scripts/compile_results.py --prefix C2DTI_EVAL_` after runs complete.

3. Publish-ready summary tables:
- Add mean and standard deviation metrics over seeds for each dataset/split/ablation.
- Add ranking by primary metric and include confidence intervals where available.

4. Causal ablation interpretation:
- Compare full versus no_xview/no_mas/no_irm/no_cf to quantify each pillar contribution.

## Conclusion

Implementation validation is strong and consistent with the 4-pillar roadmap. The remaining work is benchmark execution and statistical aggregation, not architecture wiring.