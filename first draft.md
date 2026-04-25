# C2DTI Paper Draft

## Title (Draft)

C2DTI: A Four-Pillar Causal Framework for Drug-Target Interaction Prediction with Cross-View Agreement, Masked Self-Supervision, and Invariant Counterfactual Learning

## Abstract (Draft)

Drug-target interaction prediction benefits from strong representation learning, but many models remain vulnerable to shortcut correlations and weak cross-view consistency. We present a a cross-view causal drug-target interaction prediction with multimodal causal consistency (C2DTI), a four-pillar causal framework that integrates dual-view representation modeling, cross-view causal agreement, masked autoencoder self-supervision, and IRM-plus-counterfactual regularization within a unified training objective. The core idea is  agreement objective that enforces consistency between sequence and graph evidence under perturbation.The framework is organized as a reproducible, configuration-driven pipeline with explicit ablation controls for each pillar. In implementation validation, all causal modules execute end-to-end and expose stable diagnostics for agreement, reconstruction, invariance, and counterfactual sensitivity. The evaluation matrix tooling supports large-scale comparisons across datasets, split regimes, ablations, and random seeds. Full benchmark results are generated through the matrix pipeline and compiled into detailed and aggregate reports for publication analysis.

## 1. Introduction (Draft)

Predicting drug-target interactions (DTIs) is central to drug discovery, repurposing, and safety screening. While deep learning has improved predictive performance, many DTI models overfit dataset-specific patterns and underperform in out-of-distribution scenarios such as cold-drug and cold-target splits. This creates a need for methods that are both accurate and causally robust.

C2DTI addresses this with four coordinated pillars:
1. Dual backbone representations that encode complementary views.
2. Cross-view causal agreement under perturbation.
3. Masked autoencoder self-supervision to preserve modality structure.
4. Invariant risk minimization and counterfactual rejection to reduce shortcut learning.

The key contribution is not only each pillar individually, but their integration under a unified, ablation-ready objective compatible with practical benchmark workflows.

## 2. Method (Draft)

### 2.1 Pillar 1: Dual Backbone Views

C2DTI combines representation paths for drug and target modalities and supports sequence-view features alongside frozen embedding usage. This establishes complementary perspectives for interaction scoring.

### 2.2 Pillar 2: Cross-View Causal Agreement

Cross-view consistency is enforced through agreement terms between sequence and graph predictions, including perturbation-aware terms. The objective penalizes disagreement and yields a bounded causal agreement score.

### 2.3 Pillar 3: Masked Autoencoder Self-Supervision

For each modality embedding space, a masked reconstruction objective encourages structurally informative representations. Lower reconstruction loss corresponds to stronger recoverable signal.

### 2.4 Pillar 4: IRM and Counterfactual Objectives

Invariant risk minimization reduces environment-specific error variance, while counterfactual target swaps penalize shortcut-positive predictions. Together they promote invariance and pair-specific discrimination.

### 2.5 Unified Objective

The unified scorer combines all pillar losses with configurable lambda weights:
- lambda_xview
- lambda_mas
- lambda_irm
- lambda_cf

This directly supports ablation studies by setting selected lambdas to zero while preserving a single training interface.

## 3. Experimental Protocol (Draft)

### 3.1 Datasets

Primary benchmark families include DAVIS, KIBA, and BindingDB variants as configured in repository pipelines.

### 3.2 Split Regimes

Evaluation includes random, cold-drug, and cold-target splits to measure both in-distribution and transfer behavior.

### 3.3 Matrix Design

The Phase-6 matrix script enumerates dataset x split x ablation x seed combinations and generates execution-ready configs and command sheets.

### 3.4 Metrics

Planned reporting metrics include CI, RMSE, Pearson, and Spearman, together with causal diagnostics from the unified objective.

## 4. Current Implementation Evidence (Draft)

Notebook-validated evidence currently confirms:
1. Backbone module behavior and deterministic sequence encoding.
2. Causal objective module validity and bounded scores.
3. IRM and counterfactual helper correctness.
4. Unified scorer end-to-end execution and ablation responsiveness.

Representative diagnostic values are available in notebook outputs and summarized in `docs/results_analysis.md`.

## 5. Planned Results Section (To Fill After Matrix Runs)

Add the following after full matrix execution:
1. Main performance table across datasets and splits.
2. Ablation table for full/no_xview/no_mas/no_irm/no_cf.
3. Seed-robust aggregate statistics (mean and standard deviation).
4. Error analysis for cold-start regimes.

## 6. Discussion (Draft)

Expected findings to examine:
1. Whether cross-view agreement improves robustness in cold splits.
2. Whether MAS contributes stable gains versus objective-only variants.
3. Whether IRM and counterfactual terms reduce shortcut behavior without harming base accuracy.

Potential limitations:
1. Runtime cost of full matrix evaluation.
2. Sensitivity to lambda tuning.
3. Dataset-specific imbalance effects.

## 7. Reproducibility Notes

Reproducibility is supported by:
1. Config-driven objective and model modes.
2. Scripted matrix generation and result compilation.
3. Notebook-based module-level validation before long-running experiments.

## 8. Conclusion (Draft)

C2DTI provides an integrated causal DTI pipeline that is implementation-complete at module level and ready for full benchmark execution. The next milestone is filling benchmark tables from matrix runs and finalizing quantitative claims for submission.

## Appendix A: Execution Checklist

1. Run evaluation matrix with execute mode.
2. Compile detailed and aggregate CSV reports.
3. Populate results tables and ablation figures.
4. Finalize discussion with empirical findings.
5. Lock reproducibility appendix with exact config references.