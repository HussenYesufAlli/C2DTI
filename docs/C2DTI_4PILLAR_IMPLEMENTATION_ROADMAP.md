# C2DTI 4-Pillar Implementation Roadmap -> Top Journal Paper

## Status Snapshot (2026-04-25)

This roadmap now reflects the current repository implementation state.

Current state summary:
1. Core pillar modules are implemented in the existing runtime path (`runner.py` / `binary_runner.py` / `causal_runtime.py` / `unified_scorer.py`).
2. Regression evaluation matrix is implemented and executed at 135 runs (3 datasets x 3 splits x 5 ablations x 3 seeds).
3. Binary evaluation matrix is implemented and executed at 27 runs (3 datasets x 3 splits x 3 seeds).
4. Report compilation scripts exist for both regression and binary outputs.
5. Manuscript drafting is active in docs paper draft files, with v2 refinement in progress.

## Architecture Overview

The complete C2DTI model has **4 interdependent pillars**, each enforcing different causal constraints:

```
INPUTS:
  Drug SMILES  ──→ ChemBERTa (frozen) ──→ e_drug ∈ ℝ^768
  Target Seq   ──→ ANKH (frozen)       ──→ e_prot ∈ ℝ^768
  
  Drug Graph   ──→ MixHop GNN          ──→ h_graph ∈ ℝ^d
  Target Graph ──→ MixHop GNN          ──→ h_graph ∈ ℝ^d

OUTPUTS:
  p_seq   = sigmoid(α·e_drug + (1-α)·e_prot) + MLP
  p_graph = sigmoid(h_graph) + MLP
```

---

## Implementation Phases

### Phase 1: Pillar 1 — Dual Pretrained Backbone

**Goal:** Load frozen ChemBERTa + ANKH; create sequence-view encoder.

**Tasks:**

1. **Load pretrained models**
   - ChemBERTa: `ChemBERTa-77M-MLM` from HuggingFace
   - ANKH: ANKH 30M (or 82M) from OmegaLabs
   - Check MINDG_CLASSA code for exact model IDs and tokenizers

2. **Create SequenceViewEncoder class**
   ```python
   class SequenceViewEncoder(nn.Module):
       def __init__(self, chemberta_ckpt, ankh_ckpt, alpha=0.5):
           self.drug_encoder = ChemBERTaForMaskedLM.from_pretrained(...)
           self.prot_encoder = AnkhForMaskedLM.from_pretrained(...)
           self.alpha = nn.Parameter(torch.tensor(alpha))
           self.mlp_head = nn.Sequential(
               nn.Linear(768, 256),
               nn.ReLU(),
               nn.Linear(256, 1),
               nn.Sigmoid()
           )
       
       def forward(self, drug_ids, target_ids):
           e_drug = self.drug_encoder(drug_ids).last_hidden_state[:, 0, :]  # CLS token
           e_prot = self.prot_encoder(target_ids).last_hidden_state[:, 0, :]
           fused = self.alpha * e_drug + (1 - self.alpha) * e_prot
           return self.mlp_head(fused)
   ```

3. **Freeze weights** (backprop only through MLP and alpha)

4. **Create dataset pipeline** to tokenize SMILES/sequences on the fly

**Files to create/modify:**
- `src/c2dti/backbones.py` — New file for `SequenceViewEncoder`
- `src/c2dti/dataset_loader.py` — Add tokenization support

**Unit tests:** Load models, forward pass, check output shape (batch_size, 1)

---

### Phase 2: Pillar 2 — Cross-View Causal Agreement

**Goal:** Implement perturbation + cross-view consistency loss.

**Formula:**
$$L_{XVIEW} = \text{MSE}(p_{seq}, p_{graph}) + \text{MSE}(p_{seq\_pert}, p_{graph}) + \text{MSE}(p_{seq}, p_{graph\_pert})$$

**Tasks:**

1. **Implement perturbation functions**
   ```python
   def perturb_embeddings(e_drug, e_prot, mask_ratio=0.15, seed=None):
       """Mask 15% of dimensions in drug and protein embeddings."""
       if seed: torch.manual_seed(seed)
       mask = torch.rand(e_drug.shape) < mask_ratio
       e_drug_pert = e_drug.clone()
       e_drug_pert[mask] = 0
       e_prot_pert = e_prot.clone()
       e_prot_pert[mask] = 0
       return e_drug_pert, e_prot_pert
   
   def perturb_graph_edges(adj, drop_ratio=0.1, seed=None):
       """Drop 10% of edges in adjacency matrix."""
       if seed: torch.manual_seed(seed)
       mask = torch.rand(adj.shape) < drop_ratio
       adj_pert = adj.clone()
       adj_pert[mask] = 0
       return adj_pert
   ```

2. **Create CausalConsistencyLoss class**
   ```python
   class CrossViewCausalLoss(nn.Module):
       def __init__(self, lambda_xview=0.1):
           self.lambda_xview = lambda_xview
       
       def forward(self, p_seq, p_graph, p_seq_pert, p_graph_pert):
           """
           p_seq: predictions from sequence view (unperturbed)
           p_graph: predictions from graph view (unperturbed)
           p_seq_pert: predictions from perturbed sequence embeddings
           p_graph_pert: predictions from perturbed graph
           """
           mse = nn.MSELoss()
           loss = (
               mse(p_seq, p_graph) +           # Agreement on original
               mse(p_seq_pert, p_graph) +      # Seq robustness to own perturbation
               mse(p_seq, p_graph_pert)        # Graph robustness to own perturbation
           )
           return self.lambda_xview * loss
   ```

3. **Integrate into training loop**
   - Compute unperturbed predictions
   - Compute perturbed embeddings/graphs
   - Compute perturbed predictions
   - Add cross-view loss to total objective

**Files to create/modify:**
- `src/c2dti/causal_objective.py` — Implement real cross-view loss (replace placeholder)
- `src/c2dti/causal_runtime.py` — Runtime orchestration for multi-component causal outputs

**Unit tests:** Check perturbation changes predictions, loss decreases over iterations

---

### Phase 3: Pillar 3 — Masked AutoEncoder (MAS) per Modality

**Goal:** Reconstruction loss ensures representations capture modality-specific information.

**Formula:**
$$L_{MAS\_drug} = \text{MSE}(\text{reconstruct}(e\_drug[M]), e\_drug[M])$$
$$L_{MAS\_prot} = \text{MSE}(\text{reconstruct}(e\_prot[M]), e\_prot[M])$$

**Tasks:**

1. **Create MASHead class**
   ```python
   class MASHead(nn.Module):
       def __init__(self, hidden_dim=768, mask_ratio=0.15):
           self.mask_ratio = mask_ratio
           self.decoder = nn.Sequential(
               nn.Linear(hidden_dim, 512),
               nn.ReLU(),
               nn.Linear(512, hidden_dim)
           )
       
       def forward(self, e):
           """Mask 15% of dims, reconstruct."""
           mask = torch.rand(e.shape) < self.mask_ratio
           e_masked = e.clone()
           e_masked[mask] = 0
           e_recon = self.decoder(e_masked)
           loss = torch.mean((e_recon - e)[mask] ** 2)
           return loss
   ```

2. **Add to SequenceViewEncoder**
   ```python
   class SequenceViewEncoder(nn.Module):
       def __init__(self, ...):
           ...
           self.mas_head_drug = MASHead()
           self.mas_head_prot = MASHead()
       
       def forward(self, drug_ids, target_ids, compute_mas=False):
           e_drug = ...
           e_prot = ...
           ...
           if compute_mas:
               mas_loss_drug = self.mas_head_drug(e_drug)
               mas_loss_prot = self.mas_head_prot(e_prot)
               return p_seq, mas_loss_drug, mas_loss_prot
           return p_seq
   ```

3. **Integrate into training**
   - Compute MAS losses every forward pass
   - Add to total loss with weight λ_mas

**Files to modify:**
- `src/c2dti/backbones.py` — Add `MASHead` and integrate into encoder

**Unit tests:** Check masked dims are reconstructed, loss decreases

---

### Phase 4: Pillar 4 — IRM + Counterfactual

**Goal:** Enforce invariant representations across environments + reject counterfactual negatives.

**Formula:**
$$L_{IRM} = \text{Var}_E[\nabla_W \cdot L_{BCE}(p_e, y)]^2$$
$$L_{CF} = \text{BCE}(p_{cf}, y=0) \text{ for counterfactual pairs}$$

**Tasks:**

1. **Create IRMHead class**
   ```python
   class IRMHead(nn.Module):
       def __init__(self, lambda_irm=1.0, warmup_epochs=10):
           self.lambda_irm = lambda_irm
           self.warmup_epochs = warmup_epochs
       
       def forward(self, logits, labels, epoch, env_ids):
           """
           logits: model outputs
           labels: true DTI labels
           epoch: current epoch (for warmup schedule)
           env_ids: array [0, 0, ..., 1, 1, ...] marking which environment each sample belongs
           
           Compute grad variance across environments.
           """
           bce = nn.BCEWithLogitsLoss(reduction='none')
           loss_per_sample = bce(logits, labels)
           
           # Compute gradient per environment
           env_grads = []
           for env in torch.unique(env_ids):
               mask = env_ids == env
               env_loss = loss_per_sample[mask].mean()
               env_grad = torch.autograd.grad(
                   env_loss, 
                   logits, 
                   retain_graph=True, 
                   create_graph=True
               )[0]
               env_grads.append(env_grad.mean())
           
           # Variance of gradients across environments
           grad_var = torch.var(torch.stack(env_grads))
           
           # Warmup schedule: gradually increase penalty
           warmup_weight = min(1.0, epoch / self.warmup_epochs)
           return warmup_weight * self.lambda_irm * grad_var
   ```

2. **Counterfactual data augmentation**
   ```python
   def create_counterfactual_pairs(dataset, num_cf_pairs=1000):
       """
       For each drug d with positive target t:
       - Find random different target t'
       - Create (d, t', label=0) as hard negative
       """
       cf_pairs = []
       for i, (drug, target, y) in enumerate(dataset):
           if y > 0.5:  # Positive pair
               random_idx = np.random.randint(0, len(dataset))
               _, random_target, _ = dataset[random_idx]
               cf_pairs.append((drug, random_target, 0))  # Hard negative
       return cf_pairs
   ```

3. **Create counterfactual loss**
   ```python
   class CounterfactualLoss(nn.Module):
       def __init__(self, lambda_cf=0.1):
           self.lambda_cf = lambda_cf
       
       def forward(self, p_cf, y_cf=0):
           """Enforce model rejects counterfactual pairs."""
           bce = nn.BCEWithLogitsLoss()
           return self.lambda_cf * bce(p_cf, torch.zeros_like(p_cf))
   ```

4. **Integrate into training**
   - Track environments (seq-view vs graph-view)
   - Mix CF pairs into each batch
   - Compute IRM + CF losses

**Files to create/modify:**
- `src/c2dti/irm_loss.py` — New file for IRM and CF losses
- `src/c2dti/dataset_loader.py` — Add CF augmentation

**Unit tests:** Check IRM penalty increases with env divergence, CF loss decreases

---

### Phase 5: Complete Training Pipeline

**Goal:** Unify all 4 pillars into one end-to-end training loop.

**Total Loss:**
$$L_{total} = L_{BCE} + \lambda_{xview} \cdot L_{XVIEW} + \lambda_{mas} \cdot (L_{MAS\_drug} + L_{MAS\_prot}) + \lambda_{irm} \cdot L_{IRM} + \lambda_{cf} \cdot L_{CF}$$

**Tasks:**

1. **Create UnifiedC2DTIModel class**
   ```python
   class UnifiedC2DTIModel(nn.Module):
       def __init__(self, config):
           self.seq_view = SequenceViewEncoder(...)
           self.graph_view = MixHopPredictor(...)
           self.cross_view_loss = CrossViewCausalLoss(config.lambda_xview)
           self.irm_loss = IRMHead(config.lambda_irm)
           self.cf_loss = CounterfactualLoss(config.lambda_cf)
       
       def forward(self, batch, compute_all_losses=True):
           p_seq = self.seq_view(batch['drug_ids'], batch['target_ids'])
           p_graph = self.graph_view(...)
           
           loss_bce = bce(p_seq + p_graph, batch['labels'])  # Ensemble prediction
           
           if compute_all_losses:
               # Perturb
               p_seq_pert, p_graph_pert = ...
               loss_xview = self.cross_view_loss(p_seq, p_graph, p_seq_pert, p_graph_pert)
               loss_mas_drug, loss_mas_prot = self.seq_view(..., compute_mas=True)
               loss_irm = self.irm_loss(...)
               loss_cf = self.cf_loss(...)
               
               total_loss = (loss_bce + loss_xview + loss_mas_drug + loss_mas_prot + 
                           loss_irm + loss_cf)
           else:
               total_loss = loss_bce
           
           return {
               'total': total_loss,
               'bce': loss_bce,
               'xview': loss_xview if compute_all_losses else 0,
               'mas': (loss_mas_drug + loss_mas_prot) if compute_all_losses else 0,
               'irm': loss_irm if compute_all_losses else 0,
               'cf': loss_cf if compute_all_losses else 0
           }
   ```

2. **Training loop**
   ```python
   def train_c2dti(model, train_loader, config):
       optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
       
       for epoch in range(config.epochs):
           for batch in train_loader:
               optimizer.zero_grad()
               losses = model(batch, compute_all_losses=True)
               losses['total'].backward()
               optimizer.step()
               
               # Log all components
               print(f"Epoch {epoch}: BCE={losses['bce']:.4f}, "
                     f"XVIEW={losses['xview']:.4f}, "
                     f"MAS={losses['mas']:.4f}, "
                     f"IRM={losses['irm']:.4f}, "
                     f"CF={losses['cf']:.4f}")
   ```

**Files to create/modify:**
- `src/c2dti/runner.py` — Main regression runtime entry path
- `src/c2dti/binary_runner.py` — Main binary runtime entry path
- `src/c2dti/unified_scorer.py` — Unified multi-pillar scoring path

---

### Phase 6: Comprehensive Evaluation

**Regression Evaluation Matrix:**
```
3 datasets (DAVIS, KIBA, BindingDB)
× 3 split strategies (random, cold_drug, cold_target)
× 5 ablations (full, -causal, -irm, -cf, -mas)
× 3 seeds (10, 34, 42)
= 135 total runs
```

**Binary Evaluation Matrix:**
```
3 datasets (DAVIS, KIBA, BindingDB)
× 3 split strategies (random, cold_drug, cold_target)
× 3 seeds (10, 34, 42)
= 27 total runs
```

**Baselines to compare:**
- DrugBAN
- GraphDTA
- DeepDTA
- MGraphDTA
- MINDG

**Metrics:** CI (primary), RMSE, Pearson, Spearman

**Ablation config generation path:**
- Base config: `configs/davis_unified_causal_gate.yaml`
- Generated regression matrix configs: `configs/generated_eval_matrix/*.yaml`
- Ablations are applied by lambda control (`lambda_xview`, `lambda_irm`, `lambda_cf`, `lambda_mas`) in generated configs.
- Base binary config: `configs/davis_binary_baseline.yaml`
- Generated binary matrix configs: `configs/generated_binary_eval_matrix/*.yaml`

**Files to create/modify:**
- `scripts/run_eval_matrix.py` — Automated regression evaluation pipeline
- `scripts/run_binary_eval_matrix.py` — Automated binary evaluation pipeline
- `scripts/compile_results.py` — Aggregate regression results into tables
- `scripts/compile_binary_results.py` — Aggregate binary results into tables

---

### Phase 7: Paper Writing

**Structure (for Bioinformatics / Briefings in Bioinformatics):**

1. **Introduction** (1 page)
   - DTI prediction importance
   - Causal reliability motivation
   - Paper contributions (4 pillars)

2. **Methods** (3 pages)
   - Pillar 1: Pretrained encoders
   - Pillar 2: Cross-view agreement
   - Pillar 3: MAS regularization
   - Pillar 4: IRM + Counterfactual

3. **Results** (3 pages)
   - Main evaluation matrix (3 datasets × 3 splits)
   - Ablation results
   - Robustness under perturbation
   - Comparison vs baselines

4. **Discussion** (2 pages)
   - Why each pillar matters
   - Causal reliability insights
   - Limitations
   - Future work

5. **Supplement**
   - Full ablation tables
   - Hyperparameter sensitivity
   - CI distributions (violin plots)

---

## Implementation Schedule

```
Week 1-2:  Phase 1 (Backbone loading) + Phase 2 (Cross-view)
Week 3:    Phase 3 (MAS) + Phase 4 (IRM + CF)
Week 4:    Phase 5 (Training pipeline integration)
Week 5:    Phase 6 (Evaluation matrix)
Week 6:    Phase 7 (Paper writing + revision)
```

---

## Key Hyperparameters to Tune

| Parameter | Range | Suggested |
|-----------|-------|-----------|
| `lambda_xview` | [0.01, 0.1, 0.5] | 0.1 |
| `lambda_mas` | [0.01, 0.1, 0.5] | 0.1 |
| `lambda_irm` | [1.0, 10.0, 100.0] | 10.0 |
| `lambda_cf` | [0.01, 0.1, 0.5] | 0.1 |
| `alpha` (fusion weight) | [0.3, 0.5, 0.7] | learnable |
| `mask_ratio` | [0.1, 0.15, 0.2] | 0.15 |
| `drop_ratio` (graph) | [0.05, 0.1, 0.15] | 0.1 |
| `warmup_epochs` (IRM) | [5, 10, 20] | 10 |

---

## Success Criteria for Top Journal (Updated)

✅ **Must have:**
- All 4 pillars implemented and integrated in one reproducible runtime path
- Regression (135) and binary (27) matrix runs completed and compiled
- Full result tables report uncertainty across seeds (mean +/- std or CI)
- Reproducibility section includes exact commands, config references, and commit hash

✅ **Should have:**
- Statistical significance testing for key comparisons
- Robustness curves (for example CI vs perturbation strength)
- Representation and error-analysis visualizations

✅ **Nice to have:**
- Pre-registered experiment protocol
- Public reproducibility bundle (configs, command sheets, reports)
- Interactive dashboard for split/ablation exploration

---

## Files and Artifact Summary (Current)

```
src/c2dti/
    ├── backbones.py                    (Implemented: sequence backbones + MAS components)
    ├── causal_objective.py             (Implemented: cross-view + MAS + helpers)
    ├── irm_loss.py                     (Implemented: IRM + CF losses)
    ├── causal_runtime.py               (Implemented: causal runtime orchestration)
    ├── unified_scorer.py               (Implemented: unified 4-pillar scoring)
    ├── runner.py                       (Implemented: regression runtime)
    ├── binary_runner.py                (Implemented: binary runtime)
    ├── dataset_loader.py               (Implemented: CSV loaders + fallback handling)
  └── ...existing files...

scripts/
    ├── run_eval_matrix.py              (Implemented: 135-run regression automation)
    ├── run_binary_eval_matrix.py       (Implemented: 27-run binary automation)
    ├── compile_results.py              (Implemented: regression aggregation)
    ├── compile_binary_results.py       (Implemented: binary aggregation)
  └── ...existing scripts...

configs/
    ├── davis_unified_causal_gate.yaml      (Base unified regression config)
    ├── davis_binary_baseline.yaml          (Base binary config)
    ├── generated_eval_matrix/*.yaml        (Generated regression matrix configs)
    ├── generated_binary_eval_matrix/*.yaml (Generated binary matrix configs)
  └── ...existing configs...

docs/
    ├── C2DTI_4PILLAR_IMPLEMENTATION_ROADMAP.md  (This file, status-updated)
    ├── results_analysis.md             (Implemented and active)
    ├── paper_draft.md                  (Implemented and active)
    └── paper_drafr_v2.md               (Active v2 manuscript refinement)
```
