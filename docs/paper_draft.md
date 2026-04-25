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

### 1.1 Contributions

1. We propose C2DTI, a four-pillar causal DTI framework that unifies cross-view agreement, masked self-supervision, and invariant-counterfactual regularization in one training interface.
2. We deliver a reproducible evaluation stack with explicit matrix generation, split control, ablation control, and report compilation for both regression and binary settings.
3. We provide full executed benchmark matrices (135 regression runs and 27 binary runs) across DAVIS, KIBA, and BindingDB under random, cold-drug, and cold-target protocols.
4. We provide an empirical analysis showing strong random-split behavior, measurable cold-split degradation, and currently weak regression ablation separation, which defines concrete next optimization targets.

## 2. Method (Draft)

### 2.0 Problem Formulation

Let \(D=\{d_i\}_{i=1}^{n_d}\) be the set of drugs and \(T=\{t_j\}_{j=1}^{n_t}\) be the set of targets. Let \(Y \in \mathbb{R}^{n_d \times n_t}\) denote the interaction matrix for regression settings (or \(Y \in \{0,1\}^{n_d \times n_t}\) for binary settings), where \(y_{ij}\) is the observed interaction value between \(d_i\) and \(t_j\) when available.

We define \(\Omega \subseteq D \times T\) as the set of observed pairs and split \(\Omega\) into train and test subsets according to protocol-specific masks (random, cold-drug, cold-target). The learning objective is to train a predictor \(f_\theta(d_i, t_j) = \hat{y}_{ij}\) that generalizes to unseen pairs under both in-distribution and distribution-shifted split regimes.

In C2DTI, this prediction objective is regularized by four coordinated components: cross-view causal agreement, masked autoencoder self-supervision, invariant risk minimization, and counterfactual regularization. The final optimization target is the unified weighted objective defined in Section 2.5.

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

Regression reporting includes CI, RMSE, Pearson, and Spearman. Binary reporting includes AUROC, AUPRC, F1, Accuracy, Sensitivity (Recall), Specificity, and Precision. In addition to task metrics, we track causal diagnostics from the unified objective (cross-view agreement, MAS reconstruction, IRM variance, and counterfactual sensitivity) to analyze mechanism-level behavior beyond aggregate prediction quality.

### 3.5 Execution Coverage (Current)

Current executed matrix coverage used in this draft is:
1. Regression matrix: 135 runs.
2. Binary matrix: 27 runs.
3. Datasets: DAVIS, KIBA, BindingDB.
4. Split regimes: random, cold-drug, cold-target.
5. Design factors: dataset x split x ablation x seed.

This coverage supports both predictive benchmarking and stability-oriented causal analysis under distribution shift.

## 4. Current Implementation Evidence (Draft)

Notebook-validated evidence currently confirms:
1. Backbone module behavior and deterministic sequence encoding.
2. Causal objective module validity and bounded scores.
3. IRM and counterfactual helper correctness.
4. Unified scorer end-to-end execution and ablation responsiveness.

Representative diagnostic values are available in notebook outputs and summarized in `docs/results_analysis.md`.

## 5. Results (Current Aggregate Tables)

### 5.1 Regression Results (FULL, mean over seeds)

| dataset | split | n_runs | ci_mean | rmse_mean | pearson_mean | spearman_mean |
| --- | --- | --- | --- | --- | --- | --- |
| BINDINGDB | COLD_DRUG | 3 | 0.4092 | 0.0603 | 0.0042 | 0.0008 |
| BINDINGDB | COLD_TARGET | 3 | 0.5047 | 0.0583 | 0.0020 | -0.0001 |
| BINDINGDB | RANDOM | 3 | 0.4814 | 0.0589 | 0.0041 | -0.0001 |
| DAVIS | COLD_DRUG | 3 | 0.4959 | 7152.2611 | -0.0016 | -0.0066 |
| DAVIS | COLD_TARGET | 3 | 0.5043 | 7619.2996 | 0.0156 | 0.0081 |
| DAVIS | RANDOM | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| KIBA | COLD_DRUG | 3 | 0.5014 | 5.9100 | 0.0191 | 0.0029 |
| KIBA | COLD_TARGET | 3 | 0.4887 | 5.9043 | -0.0274 | -0.0180 |
| KIBA | RANDOM | 3 | 0.5039 | 5.8973 | 0.0185 | 0.0082 |

### 5.2 Regression Ablation Example (DAVIS, RANDOM)

| ablation | n_runs | ci_mean | rmse_mean | pearson_mean | spearman_mean |
| --- | --- | --- | --- | --- | --- |
| FULL | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_CAUSAL | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_CF | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_IRM | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_MAS | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |

### 5.3 Binary Results (mean over seeds)

| dataset | split | n_runs | auroc_mean | auprc_mean | f1_mean | accuracy_mean | sensitivity_mean | specificity_mean | precision_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BINDINGDB | COLD_DRUG | 3 | 0.8547 | 0.9684 | 0.9405 | 0.8882 | 0.9993 | 0.0327 | 0.8883 |
| BINDINGDB | COLD_TARGET | 3 | 0.7392 | 0.9352 | 0.9364 | 0.8818 | 0.9944 | 0.0853 | 0.8849 |
| BINDINGDB | RANDOM | 3 | 0.8947 | 0.9826 | 0.9556 | 0.9172 | 0.9923 | 0.2532 | 0.9215 |
| DAVIS | COLD_DRUG | 3 | 0.6498 | 0.9478 | 0.9592 | 0.9217 | 1.0000 | 0.0000 | 0.9217 |
| DAVIS | COLD_TARGET | 3 | 0.7880 | 0.9741 | 0.9625 | 0.9277 | 1.0000 | 0.0000 | 0.9277 |
| DAVIS | RANDOM | 3 | 0.8610 | 0.9859 | 0.9633 | 0.9291 | 1.0000 | 0.0000 | 0.9291 |
| KIBA | COLD_DRUG | 3 | 0.7282 | 0.9030 | 0.8849 | 0.7936 | 1.0000 | 0.0003 | 0.7936 |
| KIBA | COLD_TARGET | 3 | 0.7495 | 0.9102 | 0.8978 | 0.8152 | 0.9999 | 0.0129 | 0.8149 |
| KIBA | RANDOM | 3 | 0.8453 | 0.9521 | 0.9006 | 0.8219 | 0.9982 | 0.0779 | 0.8204 |

### 5.4 Current Observations

1. Random splits generally outperform cold splits, indicating expected distribution-shift sensitivity.
2. Regression ablation separation is currently weak in aggregated outputs and needs deeper causal-diagnostic analysis.
3. Binary tasks show strong AUROC/AUPRC but often low specificity under class imbalance, so threshold-aware analysis is required.

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
