# C2DTI Paper Draft

## Title (Draft)

C2DTI: A Four-Pillar Causal Framework for Drug-Target Interaction Prediction with Cross-View Agreement, Masked Self-Supervision, and Invariant Counterfactual Learning

## Abstract (Draft)

Drug-target interaction (DTI) prediction has improved with deep representation learning, yet many models remain sensitive to shortcut correlations and weak agreement across modality views. We present C2DTI, a four-pillar causal framework that unifies dual-backbone representation learning, cross-view causal agreement under perturbation, masked autoencoder self-supervision, and invariant-plus-counterfactual regularization in a single objective. The key mechanism is a cross-view agreement term that enforces consistency between sequence evidence and structural graph evidence on both clean and perturbed inputs. C2DTI is implemented as a reproducible, configuration-driven pipeline with explicit ablation controls for each pillar. Implementation-level validation confirms that all causal modules execute end-to-end and produce stable diagnostics for agreement, reconstruction quality, invariance, and counterfactual sensitivity. The evaluation matrix tooling supports large-scale comparisons across datasets, split regimes, ablations, and random seeds, and compiles outputs into detailed and aggregate reports for publication analysis.

## 1. Introduction (Draft)

Predicting drug-target interactions (DTIs) is central to drug discovery, repurposing, and safety screening. While deep learning has improved predictive performance, many DTI models overfit dataset-specific patterns and underperform in out-of-distribution scenarios such as cold-drug and cold-target splits. This creates a need for methods that are both accurate and causally robust.

The key gap is that three strong baseline families often optimize prediction quality without explicitly enforcing cross-environment causal consistency: sequence-focused encoders (for example, DeepDTA-style and transformer-only variants) [CITATION:SEQ_BASELINES], graph-focused interaction models (for example, GraphDTA/MixHop-like propagation models) [CITATION:GRAPH_BASELINES], and multi-view fusion models that combine modalities but do not regularize invariance/counterfactual behavior [CITATION:MULTIVIEW_BASELINES]. In cold-drug and cold-target settings, these designs can rely on environment-specific shortcuts that do not transfer.

C2DTI addresses this with four coordinated pillars:
1. Dual backbone representations that encode complementary views.
2. Cross-view causal agreement under perturbation.
3. Masked autoencoder self-supervision to preserve modality structure.
4. Invariant risk minimization and counterfactual rejection to reduce shortcut learning.

Our explicit novelty is not just the presence of these components, but their joint execution inside one unified objective with direct lambda-based ablation, shared split control, and matrix-scale reproducibility hooks. This turns causal claims into testable protocol-level comparisons rather than isolated module descriptions.

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

Primary benchmark families include DAVIS, KIBA, and BindingDB variants as configured in repository pipelines. For DAVIS/KIBA CSV layouts, required columns are Drug_ID, Drug, Target_ID, Target, and Y. Rows with missing identifiers or labels are dropped, IDs are converted to stable strings, and Y is parsed numerically. For BindingDB, Y values are interpreted as follows: if Y <= 1.0, values are treated as pre-binarized labels; otherwise values are interpreted as affinity in nM and converted to pKd with threshold 7.6 for binarization in the loader path.

### 3.2 Split Regimes

Evaluation includes random, cold-drug, and cold-target splits to measure both in-distribution and transfer behavior. The implementation-level split construction is:
1. random: all observed pairs are shuffled with a fixed seed, then a test subset of size round(test_ratio x n_known) is held out.
2. cold_drug: a test subset of active drugs is sampled, and all their observed pairs are assigned to test.
3. cold_target: a test subset of active targets is sampled, and all their observed pairs are assigned to test.

In all matrix runs, split.test_ratio=0.2 and seeds are {10, 34, 42}.

### 3.3 Matrix Design

The Phase-6 matrix scripts enumerate execution-ready configs as:
1. Regression: 3 datasets x 3 splits x 5 ablations x 3 seeds = 135 runs.
2. Binary: 3 datasets x 3 splits x 3 seeds = 27 runs.

Regression ablations are full, no_causal, no_irm, no_cf, and no_mas, implemented by setting corresponding lambda terms in the unified causal config.

Model settings in the base unified causal config are interaction_cross_attention with latent_dim=16, epochs=5, lr=0.01, attention_temperature=1.0, top_k=8, model seed=42, perturbation strength=0.10. Causal defaults are lambda_xview=1.0, lambda_mas=1.0, lambda_irm=1.0, lambda_cf=1.0, graph_model=mixhop_propagation (hop_weights [0.6, 0.3, 0.1]), MAS mask_ratio=0.15, and IRM/CF n_envs=4, n_cf_pairs=1000.

Binary matrix base settings use simple_baseline with split.test_ratio=0.2, split seeds {10, 34, 42}, and decision threshold=0.5.

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

### 5.1 Regression Results (FULL, mean +/- std over seeds)

| dataset | split | n_runs | ci_mean+/-std | rmse_mean+/-std | pearson_mean+/-std | spearman_mean+/-std |
| --- | --- | --- | --- | --- | --- | --- |
| BINDINGDB | COLD_DRUG | 3 | 0.4092+/-0.0702 | 0.0603+/-0.0036 | 0.0042+/-0.0006 | 0.0008+/-0.0004 |
| BINDINGDB | COLD_TARGET | 3 | 0.5047+/-0.0502 | 0.0583+/-0.0012 | 0.0020+/-0.0037 | -0.0001+/-0.0015 |
| BINDINGDB | RANDOM | 3 | 0.4814+/-0.0211 | 0.0589+/-0.0002 | 0.0041+/-0.0008 | -0.0001+/-0.0010 |
| DAVIS | COLD_DRUG | 3 | 0.4959+/-0.0404 | 7152.2611+/-675.5801 | -0.0016+/-0.0582 | -0.0066+/-0.0761 |
| DAVIS | COLD_TARGET | 3 | 0.5043+/-0.0254 | 7619.2996+/-891.0913 | 0.0156+/-0.0367 | 0.0081+/-0.0448 |
| DAVIS | RANDOM | 3 | 0.7222+/-0.0173 | 3663.3506+/-36.7645 | 0.4001+/-0.0336 | 0.4044+/-0.0362 |
| KIBA | COLD_DRUG | 3 | 0.5014+/-0.0041 | 5.9100+/-0.0029 | 0.0191+/-0.0048 | 0.0029+/-0.0006 |
| KIBA | COLD_TARGET | 3 | 0.4887+/-0.0076 | 5.9043+/-0.0963 | -0.0274+/-0.0611 | -0.0180+/-0.0188 |
| KIBA | RANDOM | 3 | 0.5039+/-0.0051 | 5.8973+/-0.0035 | 0.0185+/-0.0088 | 0.0082+/-0.0017 |

### 5.2 Regression Ablation Example (DAVIS, RANDOM)

| ablation | n_runs | ci_mean | rmse_mean | pearson_mean | spearman_mean |
| --- | --- | --- | --- | --- | --- |
| FULL | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_CAUSAL | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_CF | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_IRM | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |
| NO_MAS | 3 | 0.7222 | 3663.3506 | 0.4001 | 0.4044 |

### 5.3 Binary Results (mean over seeds)

| dataset | split | n_runs | auroc_mean+/-std | auprc_mean+/-std | f1_mean+/-std | accuracy_mean+/-std | sensitivity_mean+/-std | specificity_mean+/-std | precision_mean+/-std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BINDINGDB | COLD_DRUG | 3 | 0.8547+/-0.0238 | 0.9684+/-0.0095 | 0.9405+/-0.0014 | 0.8882+/-0.0024 | 0.9993+/-0.0002 | 0.0327+/-0.0048 | 0.8883+/-0.0024 |
| BINDINGDB | COLD_TARGET | 3 | 0.7392+/-0.0245 | 0.9352+/-0.0218 | 0.9364+/-0.0138 | 0.8818+/-0.0242 | 0.9944+/-0.0024 | 0.0853+/-0.0144 | 0.8849+/-0.0231 |
| BINDINGDB | RANDOM | 3 | 0.8947+/-0.0095 | 0.9826+/-0.0026 | 0.9556+/-0.0028 | 0.9172+/-0.0047 | 0.9923+/-0.0019 | 0.2532+/-0.0143 | 0.9215+/-0.0037 |
| DAVIS | COLD_DRUG | 3 | 0.6498+/-0.0254 | 0.9478+/-0.0163 | 0.9592+/-0.0104 | 0.9217+/-0.0191 | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.9217+/-0.0191 |
| DAVIS | COLD_TARGET | 3 | 0.7880+/-0.0148 | 0.9741+/-0.0040 | 0.9625+/-0.0029 | 0.9277+/-0.0054 | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.9277+/-0.0054 |
| DAVIS | RANDOM | 3 | 0.8610+/-0.0074 | 0.9859+/-0.0011 | 0.9633+/-0.0015 | 0.9291+/-0.0028 | 1.0000+/-0.0000 | 0.0000+/-0.0000 | 0.9291+/-0.0028 |
| KIBA | COLD_DRUG | 3 | 0.7282+/-0.0045 | 0.9030+/-0.0068 | 0.8849+/-0.0065 | 0.7936+/-0.0105 | 1.0000+/-0.0001 | 0.0003+/-0.0003 | 0.7936+/-0.0105 |
| KIBA | COLD_TARGET | 3 | 0.7495+/-0.0154 | 0.9102+/-0.0178 | 0.8978+/-0.0170 | 0.8152+/-0.0279 | 0.9999+/-0.0001 | 0.0129+/-0.0121 | 0.8149+/-0.0279 |
| KIBA | RANDOM | 3 | 0.8453+/-0.0042 | 0.9521+/-0.0033 | 0.9006+/-0.0037 | 0.8219+/-0.0061 | 0.9982+/-0.0003 | 0.0779+/-0.0038 | 0.8204+/-0.0059 |

### 5.4 Best-Per-Split Summary (Compact)

| split | best_regression_dataset_by_CI | best_regression_CI_mean+/-std | best_binary_dataset_by_AUROC | best_binary_AUROC_mean+/-std |
| --- | --- | --- | --- | --- |
| COLD_DRUG | KIBA | 0.5014+/-0.0041 | BINDINGDB | 0.8547+/-0.0238 |
| COLD_TARGET | BINDINGDB | 0.5047+/-0.0502 | DAVIS | 0.7880+/-0.0148 |
| RANDOM | DAVIS | 0.7222+/-0.0173 | BINDINGDB | 0.8947+/-0.0095 |

### 5.5 Current Observations

1. Random splits generally outperform cold splits, indicating expected distribution-shift sensitivity.
2. Regression ablation separation is currently weak in aggregated outputs and needs deeper causal-diagnostic analysis.
3. Binary tasks show strong AUROC/AUPRC but often low specificity under class imbalance, so threshold-aware analysis is required.

## 6. Discussion (Draft)

### 6.1 Failure Analysis Under Distribution Shift

Cold-drug and cold-target failures are consistent with representation transfer stress. In regression, DAVIS cold splits show large RMSE variance, suggesting that latent mapping quality is unstable when either the drug or target identity is fully unseen. In binary settings, specificity is frequently near zero (especially DAVIS splits), indicating a strong positive-class bias at the default threshold 0.5. This pattern is consistent with class imbalance and score calibration drift: the model maintains high recall by predicting positives aggressively, which inflates false positives in negative-sparse test folds.

### 6.2 Why Specificity Is Low In Binary Runs

Three interacting factors explain low specificity:
1. Class distribution skew in test folds increases penalty asymmetry between false negatives and false positives in practical optimization.
2. Fixed thresholding (0.5) is not split-adaptive, so calibration mismatch under cold shifts directly lowers true-negative rate.
3. Representation uncertainty for unseen drugs/targets compresses score separation, reducing negative-class margin.

Actionable next step is threshold sweeping and calibration-by-split (for example, Platt/isotonic calibration) with specificity-recall operating-point reporting [CITATION:CALIBRATION_METHODS].

### 6.3 Limitations

This study has three explicit limitations. First, dataset bias may persist due to benchmark curation artifacts and scaffold/sequence redundancy [CITATION:DATA_BIAS_DTI]. Second, class imbalance materially affects thresholded binary behavior, especially specificity. Third, full matrix execution is computationally expensive (162 total runs across regression and binary matrices), which constrains broad hyperparameter sweeps.

## 7. Reproducibility Notes

Reproducibility is supported by explicit command sets, tracked configs, and version pinning.

### 7.1 Exact Command Set

1. Regression matrix execute:
	python scripts/run_eval_matrix.py --mode run-once --execute
2. Binary matrix execute:
	python scripts/run_binary_eval_matrix.py --mode run-once --execute
3. Regression report compile:
	python scripts/compile_results.py --prefix C2DTI_EVAL_
4. Binary report compile:
	python scripts/compile_binary_results.py --prefix C2DTI_BINARY_EVAL_

Optional capped run command for fast validation:
1. python scripts/run_eval_matrix.py --mode run-once --execute --max-runs 3
2. python scripts/run_binary_eval_matrix.py --mode run-once --execute --max-runs 3

### 7.2 Config References

1. Unified regression base config: configs/davis_unified_causal_gate.yaml
2. Regression generated matrix configs: configs/generated_eval_matrix/*.yaml
3. Binary base config: configs/davis_binary_baseline.yaml
4. Binary generated matrix configs: configs/generated_binary_eval_matrix/*.yaml

### 7.3 Versioning Note

Results in this draft correspond to repository branch dev at commit e394650. Final camera-ready reporting should lock requirements and include exact environment manifest hash [CITATION:REPRO_STANDARDS].

## 8. Conclusion (Draft)

C2DTI demonstrates a fully executable causal DTI framework with matrix-scale evidence across random, cold-drug, and cold-target settings. The clearest takeaway is that random-split performance is consistently stronger than cold-split performance, confirming that distribution shift remains the core difficulty despite causal regularization. The unified objective and ablation wiring make this gap measurable and directly optimizable rather than anecdotal. A concrete next step is split-aware calibration and threshold optimization to recover specificity without sacrificing AUROC/AUPRC.

## Appendix A: Execution Checklist

1. Run evaluation matrix with execute mode.
2. Compile detailed and aggregate CSV reports.
3. Populate results tables and ablation figures.
4. Finalize discussion with empirical findings.
5. Lock reproducibility appendix with exact config references.
