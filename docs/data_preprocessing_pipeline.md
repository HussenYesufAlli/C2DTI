# C2DTI Data Preprocessing & Feature Engineering Pipeline

## Overview
Complete walkthrough from raw dataset loading to model-ready data. This pipeline is dataset-agnostic (works for DAVIS, KIBA, BindingDB).

---

## Stage 1: Data Loading

**Purpose**: Load raw datasets into standardized format

**Input**: Raw dataset files
- DAVIS/KIBA: Flat CSV with columns `Drug_ID, Drug, Target_ID, Target, Y`
- BindingDB: CSV with columns `Drug_ID, Drug, Target_ID, Target, Label` or `Y`

**Process** (`src/c2dti/dataset_loader.py`):
```python
from src.c2dti.dataset_loader import DAVISLoader, KIBALoader, BindingDBLoader
from pathlib import Path

# Load any dataset
loader = DAVISLoader(Path('data/davis'))
dataset = loader.load()

# Result: DTIDataset object with:
print(len(dataset.drugs))           # 68 (SMILES strings)
print(len(dataset.targets))         # 379 (protein sequences)
print(dataset.interactions.shape)   # (68, 379) - affinity matrix
print(dataset.metadata)             # {'source': 'DAVIS', 'n_drugs': 68, ...}
```

**Output**: `DTIDataset` object containing:
- `drugs`: List of SMILES strings (e.g., `['Cc1ccc(NC(=O)...', 'NCCCCCC...', ...]`)
- `targets`: List of protein sequences (e.g., `['MKKFFD...', 'MTVKTE...', ...]`)
- `interactions`: Numpy array (n_drugs × n_targets) with affinity values
- `metadata`: Dictionary with dataset info

---

## Stage 2: Data Cleaning

**Purpose**: Validate, normalize, and clean raw data

**Automatic Cleaning** (in loader):

1. **Column validation**: Check required columns exist
   ```python
   required = {"Drug_ID", "Drug", "Target_ID", "Target", "Y"}
   # If missing → returns placeholder dataset
   ```

2. **Type coercion**: Ensure correct data types
   ```python
   df["Drug_ID"]   = pd.to_numeric(df["Drug_ID"], errors="coerce").astype("Int64")
   df["Target_ID"] = pd.to_numeric(df["Target_ID"], errors="coerce").astype("Int64")
   df["Drug"]      = df["Drug"].astype(str)           # SMILES
   df["Target"]    = df["Target"].astype(str)        # Sequences
   df["Y"]         = pd.to_numeric(df["Y"], errors="coerce")
   ```

3. **Drop missing values**: Remove incomplete rows
   ```python
   df = df.dropna(subset=["Drug_ID", "Drug", "Target_ID", "Target", "Y"])
   ```

4. **BindingDB-specific**: Label normalization
   - Raw affinities in nM (smaller = stronger binding)
   - Pre-binarized labels (Y ≤ 1 = strong binder, Y > 1 = weak)
   - Handled transparently in loader

**Output**: Cleaned `DTIDataset` with guaranteed valid data

---

## Stage 3: Feature Engineering

**Purpose**: Convert SMILES/sequences into fixed-size numerical vectors

**Key Function**: `build_string_feature_matrix()` in `src/c2dti/data_utils.py`

```python
from src.c2dti.data_utils import build_string_feature_matrix

# Convert drug SMILES to features
smiles_list = ['Cc1ccc(NC(=O)...', 'NCCCCCC...']
drug_features = build_string_feature_matrix(smiles_list, vector_size=16)
print(drug_features.shape)  # (n_drugs, 16) - 16-dim feature vectors

# Each SMILES → 16-dimensional feature vector
# Computed by hashing each character → normalized embedding
```

**How it works**:
1. Take SMILES string: `"CCO"` (ethanol)
2. Hash each character to a position: C→pos1, C→pos2, O→pos3
3. Aggregate positions into 16-dim normalized vector
4. Result: Deterministic, normalized feature vector

**Used for**:
- Computing drug-drug similarity
- Computing target-target similarity  
- Prior embeddings in graph models

**Parameters**:
- `vector_size`: Default 16 (tradeoff: larger = more expressive, slower)

---

## Stage 4: Train/Test Split Strategy

**Purpose**: Ensure fair evaluation (avoid data leakage)

**Options** (`src/c2dti/splitter.py`):

### 4.1 Random Split
```python
from src.c2dti.splitter import RandomSplitter

splitter = RandomSplitter(test_size=0.2, random_state=42)
train_mask, test_mask = splitter.split(dataset)

# ~80% pairs for training, ~20% for testing
# Fast but unrealistic—model sees similar drugs/targets in train and test
```

### 4.2 Cold Drug Split
```python
from src.c2dti.splitter import ColdDrugSplitter

splitter = ColdDrugSplitter(test_size=0.2, random_state=42)
train_mask, test_mask = splitter.split(dataset)

# Entire drugs held out from training
# Realistic: model predicts affinity for NEW drugs
# Harder evaluation
```

### 4.3 Cold Target Split
```python
from src.c2dti.splitter import ColdTargetSplitter

splitter = ColdTargetSplitter(test_size=0.2, random_state=42)
train_mask, test_mask = splitter.split(dataset)

# Entire targets held out from training
# Realistic: model predicts affinity for NEW proteins
# Harder evaluation
```

**Output**: Boolean masks
```python
train_mask  # (n_drugs, n_targets) - True where training data
test_mask   # (n_drugs, n_targets) - True where test data
```

---

## Stage 5: Model Training

**Purpose**: Learn to predict drug-target affinity

**Two-stage process**:

### 5.1 Pre-training (Feature Learning)
```python
from src.c2dti.dti_model import MatrixFactorizationDTIPredictor

model = MatrixFactorizationDTIPredictor(
    n_drugs=68,
    n_targets=379,
    latent_dim=32,              # Dimension of learned embeddings
    learning_rate=0.01,
    reg_strength=0.001          # L2 regularization
)

# Initialize embeddings randomly
model.initialize()

# Optionally: Initialize with string features (cold-start warmup)
drug_features = build_string_feature_matrix(dataset.drugs)
target_features = build_string_feature_matrix(dataset.targets)
model.warm_start(drug_features, target_features)
```

### 5.2 Training Loop
```python
# Train on train set only
train_predictions = model.predict(dataset.interactions, train_mask)
train_loss = model.train_step(train_predictions, dataset.interactions[train_mask])

# Evaluate on test set (no backprop)
test_predictions = model.predict(dataset.interactions, test_mask)
test_loss = model.evaluate(test_predictions, dataset.interactions[test_mask])
```

**Key Models Available**:

1. **SimpleMatrixDTIPredictor** - Baseline
   - Predicts: average per drug + average per target
   - No learnable parameters
   - Fast baseline

2. **MatrixFactorizationDTIPredictor** - PRIMARY
   - Learns d-dimensional embeddings for each drug/target
   - Predicts: drug_embedding · target_embedding
   - Trainable, scalable

3. **MixHopPropagationDTIPredictor** - Graph-based
   - Uses drug-drug and target-target graphs
   - Multi-hop information propagation
   <!-- - Better for sparse data -->

4. **InteractionCrossAttentionDTIPredictor** - Advanced
   - Attention mechanism over drug-target interactions
   - Learns contextual representations
   <!-- - SOTA performance (slower) -->

---

## Stage 6: Evaluation

**Purpose**: Measure model performance

**Metrics** (`src/c2dti/evaluation.py`):

### Primary Metric: Concordance Index (CI)
```python
from src.c2dti.evaluation import compute_ci

predictions = model.predict(dataset.interactions, test_mask)
ci = compute_ci(predictions[test_mask], dataset.interactions[test_mask])

print(f"CI: {ci:.3f}")  # 0.5 = random, 1.0 = perfect ranking
```

**Interpretation**:
- **0.5**: Random predictions
- **0.7**: Good predictions
- **0.9+**: Excellent predictions
- **Metric meaning**: "Fraction of test pairs ranked correctly"

### Secondary Metrics

**MSE/RMSE** - Magnitude of errors
```python
mse = mean_squared_error(predictions, true_values)
rmse = np.sqrt(mse)
```

**Pearson Correlation** - Linear relationship
```python
pearson_r, pearson_p = pearsonr(predictions, true_values)
```

**Spearman Correlation** - Rank-based relationship
```python
spearman_r, spearman_p = spearmanr(predictions, true_values)
```

---

## Complete Pipeline Example

```python
from pathlib import Path
from src.c2dti.dataset_loader import DAVISLoader
from src.c2dti.splitter import ColdDrugSplitter
from src.c2dti.dti_model import MatrixFactorizationDTIPredictor
from src.c2dti.evaluation import compute_ci
from src.c2dti.data_utils import build_string_feature_matrix

# Stage 1: Load
dataset = DAVISLoader(Path('data/davis')).load()
print(f"Loaded: {len(dataset.drugs)} drugs, {len(dataset.targets)} targets")

# Stage 2: Clean (automatic)
print("Data cleaned automatically during loading")

# Stage 3: Features
drug_features = build_string_feature_matrix(dataset.drugs, vector_size=16)
target_features = build_string_feature_matrix(dataset.targets, vector_size=16)
print(f"Drug features: {drug_features.shape}")
print(f"Target features: {target_features.shape}")

# Stage 4: Split
splitter = ColdDrugSplitter(test_size=0.2, random_state=42)
train_mask, test_mask = splitter.split(dataset)
print(f"Train: {train_mask.sum()} pairs, Test: {test_mask.sum()} pairs")

# Stage 5: Train
model = MatrixFactorizationDTIPredictor(
    n_drugs=len(dataset.drugs),
    n_targets=len(dataset.targets),
    latent_dim=32
)
model.initialize()

for epoch in range(100):
    train_pred = model.predict(dataset.interactions, train_mask)
    loss = model.train_step(train_pred, dataset.interactions[train_mask])
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}")

# Stage 6: Evaluate
test_pred = model.predict(dataset.interactions, test_mask)
ci = compute_ci(test_pred[test_mask], dataset.interactions[test_mask])
print(f"Final CI: {ci:.3f}")
```

---

## Configuration (YAML)

All pipeline parameters in `configs/*.yaml`:

```yaml
# Dataset
dataset:
  name: davis
  split_type: cold_drug
  test_size: 0.2

# Features
features:
  vector_size: 16
  use_string_features: true

# Model
model:
  type: matrix_factorization
  latent_dim: 32
  learning_rate: 0.01
  reg_strength: 0.001
  epochs: 100

# Evaluation
evaluation:
  primary_metric: ci
  all_metrics: [mse, rmse, pearson, spearman, ci]
```

---

## Data Flow Summary

```
Raw Data (CSV)
    ↓
[Stage 1] Load → DTIDataset (drugs, targets, interactions)
    ↓
[Stage 2] Clean → Validated DTIDataset
    ↓
[Stage 3] Features → SMILES/sequences → 16-dim vectors
    ↓
[Stage 4] Split → train_mask, test_mask (80/20)
    ↓
[Stage 5] Train → Learn embeddings via gradient descent
    ↓
[Stage 6] Evaluate → CI, RMSE, Pearson, Spearman scores
    ↓
Results (metrics, predictions, model)
```

---

## Key Hyperparameters to Tune

| Parameter | Range | Default | Impact |
|-----------|-------|---------|--------|
| `latent_dim` | 8-128 | 32 | Model capacity; higher = better fit, slower |
| `learning_rate` | 0.0001-0.1 | 0.01 | Training speed; higher = faster but unstable |
| `reg_strength` | 0.0-0.1 | 0.001 | Prevent overfitting; higher = more regularization |
| `vector_size` | 8-32 | 16 | Feature expressiveness |
| `test_size` | 0.1-0.3 | 0.2 | Evaluation set fraction |

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| NaN in features | SMILES/sequences have missing values | Loader drops NaN automatically |
| Poor performance | Train/test leakage | Use ColdDrug/ColdTarget splits, not Random |
| Model not learning | Learning rate too low/high | Try [0.001, 0.01, 0.1] |
| Slow training | latent_dim too large | Reduce to 16-32 |
| Overfitting | reg_strength too low | Increase to 0.01-0.1 |

---

## References

- **CI (Concordance Index)**: Harrell et al., 1982 - Standard ranking metric in survival analysis
- **Matrix Factorization**: Koren et al., 2009 - Netflix Prize approach
- **MixHop**: Abu-El-Haija et al., 2019 - Multi-hop graph propagation
- **DTI Benchmarks**: Davis & Kuang datasets from original papers
