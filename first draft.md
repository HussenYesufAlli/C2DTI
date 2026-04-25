# C2DTI Paper Draft

## Title (Draft)

C2DTI: A Four-Pillar Causal Framework for Drug-Target Interaction Prediction with Cross-View Agreement, Masked Self-Supervision, and Invariant Counterfactual Learning

## Abstract (Draft)

Drug-target interaction (DTI) prediction has improved with deep representation learning, yet many models remain sensitive to shortcut correlations and weak agreement across modality views. We present C2DTI, a four-pillar causal framework that unifies dual-backbone representation learning, cross-view causal agreement under perturbation, masked autoencoder self-supervision, and invariant-plus-counterfactual regularization in a single objective. The key mechanism is a cross-view agreement term that enforces consistency between sequence evidence and structural graph evidence on both clean and perturbed inputs. C2DTI is implemented as a reproducible, configuration-driven pipeline with explicit ablation controls for each pillar. Implementation-level validation confirms that all causal modules execute end-to-end and produce stable diagnostics for agreement, reconstruction quality, invariance, and counterfactual sensitivity. The evaluation matrix tooling supports large-scale comparisons across datasets, split regimes, ablations, and random seeds, and compiles outputs into detailed and aggregate reports for publication analysis.

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

C2DTI starts from dual sequence encoders: ChemBERTa for drug SMILES and ANKH for protein sequences. In our current implementation, these backbones are used in frozen mode and produce modality-specific embeddings that are fused by a lightweight interaction head. This branch provides the base prediction signal and defines the sequence-view environment used by the causal objectives.

### 2.2 Pillar 2: Cross-View Causal Agreement

Pillar 2 enforces agreement between sequence-view and graph-view predictions under both clean and perturbed conditions. Let \(p_{seq}\) and \(p_{graph}\) denote predictions from the two views, and let \(p_{seq}^{pert}\), \(p_{graph}^{pert}\) denote predictions after modality-specific perturbations (token masking for sequence, structural perturbation for graph). The cross-view loss is:

$$
L_{xview} = \mathrm{MSE}(p_{seq}, p_{graph}) + \mathrm{MSE}(p_{seq}^{pert}, p_{graph}) + \mathrm{MSE}(p_{seq}, p_{graph}^{pert}).
$$

This objective discourages view-specific shortcuts and favors signals that remain stable across modalities and interventions.

### 2.3 Pillar 3: Masked Autoencoder Self-Supervision

For each modality embedding space, we mask a subset of embedding dimensions and reconstruct the masked part from the unmasked part with a lightweight decoder. This is applied independently to drug and protein embeddings:

$$
L_{MAS} = L_{MAS}^{drug} + L_{MAS}^{prot},
$$

where each term is a masked-dimension reconstruction MSE. Lower \(L_{MAS}\) indicates stronger recoverable structure, and therefore more informative latent representations.

### 2.4 Pillar 4: IRM and Counterfactual Objectives

Pillar 4 combines invariant risk minimization (IRM) with counterfactual (CF) regularization. IRM encourages a shared predictive rule across environments, which in C2DTI correspond to sequence-view and graph-view prediction contexts. CF regularization evaluates intervention-style pair perturbations (for example, target swaps) and penalizes high-confidence shortcut behavior on implausible pairs.

Together, these terms promote features that are predictive across environments while preserving pair-specific discrimination.

### 2.5 Unified Objective

All components are combined in a single training objective:

$$
L_{total} = L_{task} + \lambda_{xview}L_{xview} + \lambda_{mas}L_{MAS} + \lambda_{irm}L_{IRM} + \lambda_{cf}L_{CF}.
$$

The \(\lambda\) coefficients are configuration-controlled, so each pillar can be activated or removed for ablation without changing the training interface.

### 2.6 Figure 1: Architecture Overview (Caption Draft)

Figure 1 illustrates the full C2DTI pipeline. Drug SMILES and protein sequences are first encoded by dual frozen backbones to produce sequence-view representations (Pillar 1). In parallel, a structural branch computes graph-view predictions through MixHop-style multi-hop propagation. Cross-view causal agreement is then enforced between sequence and graph predictions under both clean and perturbed conditions (Pillar 2). On top of modality embeddings, masked autoencoder self-supervision reconstructs masked embedding dimensions for drug and protein spaces to preserve recoverable structure (Pillar 3). Finally, invariant risk minimization aligns predictive behavior across sequence-view and graph-view environments, while a counterfactual rejection term penalizes shortcut-positive responses under intervention-style pair perturbations (Pillar 4). All components are optimized through a unified objective that combines task loss with weighted causal regularizers, enabling direct ablation by adjusting lambda terms.

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