# C2DTI Data Preprocessing & Feature Engineering Pipeline

A beginner-friendly guide to understanding how data flows through the C2DTI system from raw files to model predictions.

---

## Pillar-Centric Implementation Map (Clear Placement)

This section maps each C2DTI pillar to the exact folder, file, class/function, and loss term.

### Pillar Summary

| Pillar | Purpose | Key Loss |
|---|---|---|
| Pillar 1 | Load frozen ChemBERTa + ANKH, fuse modalities | MSE between views |
| Pillar 2 | Enforce cross-view stability under perturbation | L_XVIEW (robustness core) |
| Pillar 3 | Per-modality reconstruction regularization | L_MAS_drug + L_MAS_prot |
| Pillar 4 | Invariant representations + reject false pairs | L_IRM + L_CF |

### Folder and File Placement

| Area | Folder/File | Role |
|---|---|---|
| Main runtime (continuous) | `scripts/run.py` | Entry point for regression/continuous runs |
| Main runtime (binary) | `scripts/run_binary.py` | Entry point for binary runs |
| Shared runner (continuous) | `src/c2dti/runner.py` | Regression pipeline execution |
| Shared runner (binary) | `src/c2dti/binary_runner.py` | Binary pipeline execution |
| Shared causal runtime | `src/c2dti/causal_runtime.py` | Unified causal execution for both modes |
| Causal objective functions | `src/c2dti/causal_objective.py` | Pillar loss implementations |
| Unified 4-pillar combiner | `src/c2dti/unified_scorer.py` | Combines Pillar 2/3/4 with lambdas |
| Predictive models | `src/c2dti/dti_model.py` | Predictor classes and model factory |
| Regression data loader | `src/c2dti/dataset_loader.py` | Continuous data loading |
| Binary data loader | `src/c2dti/binary_dataset_loader.py` | Binary data loading |
| Perturbation utility | `src/c2dti/perturbation.py` | Creates intervention-like perturbed data |
| IRM/CF utility | `src/c2dti/irm_loss.py` | Low-level IRM and counterfactual terms |

### Class and Function Placement by Pillar

| Pillar | Main Functions / Classes | Where |
|---|---|---|
| Pillar 1 | `DualFrozenBackbonePredictor`, `create_predictor`, `load_frozen_entity_embeddings` | `src/c2dti/dti_model.py`, `src/c2dti/backbones.py` |
| Pillar 2 | `compute_cross_view_causal_metrics`, `perturb_dataset_interactions` | `src/c2dti/causal_objective.py`, `src/c2dti/perturbation.py` |
| Pillar 3 | `compute_mas_losses`, `MASHead` | `src/c2dti/causal_objective.py`, `src/c2dti/backbones.py` |
| Pillar 4 | `compute_irm_cf_losses`, `compute_irm_penalty`, `compute_counterfactual_loss` | `src/c2dti/causal_objective.py`, `src/c2dti/irm_loss.py` |
| Unified Objective | `UnifiedC2DTIScorer.score` | `src/c2dti/unified_scorer.py` |
| Shared Execution (both modes) | `compute_causal_outputs` | `src/c2dti/causal_runtime.py` |

### Utilities and Model Notes

| Type | Item |
|---|---|
| Utility | `split_dataset` in `src/c2dti/splitter.py` supports `random`, `cold_drug`, `cold_target` |
| Utility | `evaluate_predictions` and `evaluate_binary_predictions` handle continuous vs binary metrics |
| Utility | `config_validation.py` validates model, split, and causal config schema |
| Model family | `simple_baseline`, `dual_frozen_backbone`, `matrix_factorization`, `mixhop_propagation`, `interaction_cross_attention`, `end_to_end_char_encoder` |
| Experiment outputs | summaries, predictions, and registry rows are persisted through `output_io.py` |

### Upstream Embedding Source (MINDG_CLASSA)

The frozen embeddings used in C2DTI were prepared upstream in MINDG_CLASSA:

| Script | Backbone | Output |
|---|---|---|
| `MINDG_CLASSA/scripts/esm2.py` | ESM2 | target embedding NPZ |
| `MINDG_CLASSA/scripts/proT5.py` | ProtT5 | target embedding NPZ |
| `MINDG_CLASSA/scripts/ankh.py` | ANKH | target embedding NPZ |
| `MINDG_CLASSA/scripts/drug_transformer.py` | ChemBERTa / MolFormer | drug embedding NPZ |

### Binary and Continuous Compatibility (Important)

Both run paths now use the same causal pillar engine through `compute_causal_outputs`:

- Continuous: `scripts/run.py` -> `runner.py` -> `compute_causal_outputs`
- Binary: `scripts/run_binary.py` -> `binary_runner.py` -> `compute_causal_outputs`

This keeps Pillar 2/3/4 behavior consistent across both tasks while preserving separate evaluation metrics:

- Continuous metrics: RMSE, Pearson, Spearman, CI
- Binary metrics: AUROC, AUPRC, F1, Accuracy, Sensitivity, Specificity

---

## Pipeline Overview

```
Raw Data Files
    ↓
[STAGE 1] Data Loading (dataset_loader.py)
    ↓
[STAGE 2] Data Cleaning & Validation (built into loader)
    ↓
[STAGE 3] Feature Engineering (data_utils.py)
    ↓
[STAGE 4] Train/Test Split (splitter.py)
    ↓
[STAGE 5] Model Training & Prediction (dti_model.py)
    ↓
[STAGE 6] Evaluation (evaluation.py)
    ↓
Results & Metrics
```

---

## STAGE 1: Data Loading

**File:** `src/c2dti/dataset_loader.py`

### What Happens?
Raw datasets are converted into a standardized format regardless of source.

### Three Dataset Types Supported:

#### A) DAVIS Dataset
- **Input Format**: Directory with three text files:
  - `davis.csv`: Flat CSV with columns: `Drug_ID`, `Drug` (SMILES), `Target_ID`, `Target` (sequence), `Y` (affinity)
  - Alternative format: separate text files `drug_smiles.txt`, `target_sequences.txt`, `Y.txt`
  
- **Class**: `DAVISLoader`
- **How it works**:
  1. Read CSV file
  2. Validate required columns exist
  3. Extract unique drugs and targets sorted by ID
  4. Build affinity matrix from rows

#### B) KIBA Dataset
- **Input Format**: Same structure as DAVIS
- **Class**: `KIBALoader`
- **Key difference**: Same CSV format but different affinity values

#### C) BindingDB Dataset
- **Input Format**: CSV file with columns: `Drug_ID`, `Target_ID`, `Y`
  - Optional columns: `Drug` (SMILES), `Target` (sequence)
  - May also contain: `Drug (SMILES)`, `Target (sequence)`, `Label` (normalized column names)
  
- **Class**: `BindingDBLoader`
- **Special handling**:
  - Supports multiple column name variations
  - Converts raw affinity (nM) to pKd values
  - Binarizes using threshold (default: 7.6)

### Standard Output Format

All loaders return a `DTIDataset` object:

```python
@dataclass
class DTIDataset:
    drugs: List[str]                    # Drug SMILES strings or IDs
    targets: List[str]                  # Target protein sequences or IDs
    interactions: np.ndarray            # Shape (n_drugs, n_targets), values 0-1 or continuous
    metadata: Dict[str, Any]            # Dataset-specific info (source, threshold, etc.)
```

### Example Data Flow (DAVIS):

```
Input CSV (davis.csv):
Drug_ID | Drug            | Target_ID | Target          | Y
--------|-----------------|-----------|-----------------|--------
0       | CC(C)Cc1cc...   | 0         | MQSQH...        | 0.65
1       | CCc1cc(OC)...   | 0         | MQSQH...        | 0.42
0       | CC(C)Cc1cc...   | 1         | MSTQ...         | 0.18

↓ (After loading)

DTIDataset:
  drugs = ["CC(C)Cc1cc...", "CCc1cc(OC)..."]  # 2 drugs (sorted by ID)
  targets = ["MQSQH...", "MSTQ..."]           # 2 targets (sorted by ID)
  interactions = [[0.65, 0.18],
                  [0.42, NaN]]                 # 2×2 matrix
  metadata = {source: "DAVIS", n_drugs: 2, n_targets: 2, ...}
```

### Key Functions:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `load_dti_dataset()` | Entry point | dataset_name (str), data_path (Path) | DTIDataset |
| `create_dataset_loader()` | Factory function | dataset_name, data_path | DTIDatasetLoader instance |
| `DAVISLoader.load()` | Load DAVIS | — | DTIDataset |
| `KIBALoader.load()` | Load KIBA | — | DTIDataset |
| `BindingDBLoader.load()` | Load BindingDB | — | DTIDataset |

---

## STAGE 2: Data Cleaning & Validation

**Built-in to data loaders** (Stage 1)

### What Happens?

Each loader performs automatic cleaning:

1. **Column Validation**
   - Check all required columns exist
   - Normalize column name variations (e.g., `Drug (SMILES)` → `Drug`)
   - Example: BindingDB accepts both `Y` and `Label` for affinity values

2. **Type Coercion**
   ```python
   df["Drug_ID"] = df["Drug_ID"].astype(str)           # Normalize to strings
   df["Target_ID"] = df["Target_ID"].astype(str)       # Avoid float issues
   df["Y"] = pd.to_numeric(df["Y"], errors="coerce")   # Parse numeric values
   ```

3. **Missing Data Handling**
   - Drop rows with NaN in critical columns (`Drug_ID`, `Target_ID`, `Y`)
   - Fill missing Drug/Target sequences with empty strings
   - Matrix positions with no data remain as NaN (ignored during training)

4. **Affinity Value Normalization**
   
   **For DAVIS/KIBA**: Values stored as-is (already normalized)
   
   **For BindingDB**: Raw affinities converted to pKd scale:
   ```python
   # If Y > 1.0 (looks like raw nM affinity)
   pKd = -log₁₀(Y × 10⁻⁹ + 1e-10)
   
   # Binarize using threshold (default 7.6)
   if pKd <= threshold:
       label = 1  # Binding
   else:
       label = 0  # Non-binding
   ```

5. **Deduplication**
   - Remove duplicate SMILES/sequences while keeping unique drug-target pairs
   - Maintains matrix index integrity

### Validation Done by `data_check.py`:

```python
def check_data(config_path):
    # 1. Validate config YAML syntax and required fields
    # 2. Check dataset files exist at specified paths
    # 3. Load dataset and verify matrix dimensions
    # 4. Report schema details (expected files, columns, etc.)
    # 5. Create JSON report for debugging
```

---

## STAGE 3: Feature Engineering

**File:** `src/c2dti/data_utils.py`

### What Happens?

Simple but deterministic feature representations are created from strings (SMILES/sequences).

### Function: `build_string_feature_matrix()`

**Why?** The model needs numeric features, not strings. This function converts variable-length SMILES and protein sequences into fixed-size vectors.

**Algorithm** (beginner explanation):
```
For each drug/target string:
  1. Take each character in the string
  2. Hash the character to a position (0 to vector_size-1)
     position = ASCII_value % vector_size
  3. Increment the count at that position
  4. Normalize the vector to unit length (divide by sum)

Example: SMILES "CC" with vector_size=16
  'C' (ASCII 67): position = 67 % 16 = 3
  'C' (ASCII 67): position = 67 % 16 = 3
  
  Raw counts: [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  After normalization (L2): [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Output

- **Shape**: `(n_drugs, vector_size)` or `(n_targets, vector_size)`
- **Default vector_size**: 16
- **Data type**: `float32`
- **Properties**: Deterministic (same input → same output), normalized

### Where Used?

```python
# In SimpleMatrixDTIPredictor.fit_predict():
drug_features = build_string_feature_matrix(dataset.drugs)        # (n_drugs, 16)
target_features = build_string_feature_matrix(dataset.targets)    # (n_targets, 16)

# Compute feature similarity as a lightweight prior
feature_prior = drug_features @ target_features.T  # (n_drugs, n_targets)
```

### Function: `summarize_matrix()`

**Purpose**: Compute quick statistics for logging/debugging

**Input**: Any 2D array (predictions, interactions, etc.)

**Output**:
```python
{
    "mean": float,    # Average value
    "std": float,     # Standard deviation
    "min": float,     # Minimum value
    "max": float      # Maximum value
}
```

---

## STAGE 4: Train/Test Split

**File:** `src/c2dti/splitter.py`

### What Happens?

Divide known drug-target pairs into non-overlapping train and test sets.

**Why?** The model must be evaluated on pairs it has never seen during training—otherwise metrics are artificially high (memorization, not generalization).

### Three Split Strategies:

#### A) Random Split
**What**: Randomly select ~20% of all known pairs for testing

**Use Case**: Quick experiments, smoke tests

**Implementation**:
```python
train_mask, test_mask = split_dataset(
    dataset,
    strategy="random",
    test_ratio=0.2,
    seed=42
)
# Both are boolean arrays of shape (n_drugs, n_targets)
# train_mask[i,j] = True if pair (i,j) is in training set
# test_mask[i,j] = True if pair (i,j) is in test set
```

**Data Distribution**: Random (no structure preserved)

#### B) Cold Drug Split
**What**: Hold out entire drugs (with all their targets) in test set

**Use Case**: **Most realistic**—simulates predicting for a NEW drug the model has never seen

**Difficulty**: Hard (forces generalization to novel drugs)

**Implementation**:
```python
# Example: 3 drugs total
if test_ratio=0.33:
    # Drug 0 → held out (test)
    # Drugs 1, 2 → training
    # All known pairs of drug 0 go to test_mask
    # All pairs of drugs 1,2 go to train_mask
```

#### C) Cold Target Split
**What**: Hold out entire targets (proteins) in test set

**Use Case**: Simulate predicting for a NEW protein target

**Difficulty**: Hard (forces generalization to novel targets)

**Implementation**: Mirror of cold drug—operates on targets (columns) instead of drugs (rows)

### Output

Both functions return:
```python
train_mask : np.ndarray of shape (n_drugs, n_targets), dtype=bool
test_mask : np.ndarray of shape (n_drugs, n_targets), dtype=bool
```

### Example:

```
Original interactions matrix (2 drugs, 3 targets):
[[0.7, 0.2, NaN],
 [NaN, 0.8, 0.3]]

After random split (test_ratio=0.5):
train_mask: [[True,  False, False],
             [False, True,  False]]   # 2 pairs in train

test_mask:  [[False, True,  False],
             [False, False, True]]    # 2 pairs in test
```

### Configuration

```yaml
split:
  strategy: "random"        # or "cold_drug" or "cold_target"
  test_ratio: 0.2          # 20% of known pairs go to test
  seed: 42                 # For reproducibility
```

---

## STAGE 5: Model Training & Prediction

**File:** `src/c2dti/dti_model.py`

### What Happens?

Raw dataset is transformed into dense predictions for all drug-target pairs.

### Entry Point

```python
predictor = create_predictor(model_config)
predictions = predictor.fit_predict(dataset, train_mask=train_mask)
# predictions: shape (n_drugs, n_targets), dtype float32, values [0, 1]
```

### Four Predictor Types:

#### 1) SimpleMatrixDTIPredictor (Fast Baseline)

**Idea**: Use row/column averages + feature similarity as a lightweight prior.

**Algorithm**:
```
For each pair (i, j):
  predictions[i,j] = 0.4 * row_mean[i] 
                   + 0.4 * col_mean[j]
                   + 0.1 * global_mean
                   + 0.1 * feature_similarity[i,j]

Where:
  row_mean[i] = average affinity of drug i across all targets
  col_mean[j] = average affinity of target j across all drugs
  global_mean = average affinity across entire matrix
  feature_similarity = drug_features @ target_features.T (from Stage 3)
```

**Training**: None (non-trainable)

**Speed**: Instant

**Use**: Baseline comparisons, tests

**Config**:
```yaml
model:
  name: simple_baseline
```

#### 2) MatrixFactorizationDTIPredictor (Trainable)

**Idea**: Learn a compressed representation (embedding) for each drug and target.

**Beginner Explanation**:
Think of drugs and targets as points in a low-dimensional space. Similar drugs cluster together; similar targets cluster together. The affinity between drug d and target t is the dot product (inner product) of their position vectors.

**Algorithm**:
```
1. Initialize random drug embeddings P (n_drugs × latent_dim)
2. Initialize random target embeddings Q (n_targets × latent_dim)

For each epoch (1 to epochs):
    3. Compute predictions = P @ Q.T                    # (n_drugs, n_targets)
    4. Compute error = predictions - observed_affinities (only on known pairs)
    5. Compute gradients of mean squared error (MSE)
    6. Update embeddings: P = P - lr * grad_P
    7. Update embeddings: Q = Q - lr * grad_Q
    8. Record loss for logging

8. Apply sigmoid: predictions = 1 / (1 + exp(-predictions))  # Map to [0,1]
```

**Hyperparameters** (set in config):
```yaml
model:
  name: matrix_factorization
  latent_dim: 32        # Size of embedding (default 32)
  epochs: 100           # Training iterations (default 100)
  lr: 0.01              # Learning rate (default 0.01)
  seed: 42              # Random seed (default 42)
```

**Training Requirement**: Yes (iterative optimization)

**Speed**: Slow (~seconds to minutes depending on dataset size)

**Use**: Main production model

**Key Features**:
- Stores trained embeddings (can be saved/loaded)
- Tracks loss history for debugging
- Respects `train_mask` (never peeks at test data)

#### 3) MixHopPropagationDTIPredictor (Graph-based)

**Idea**: Build drug and target similarity graphs, then propagate known interactions through multiple "hops".

**Algorithm** (simplified):
```
1. Build drug similarity graph (using cosine similarity of features)
2. Keep top-k nearest neighbors per drug
3. Build target similarity graph (same process)

4. For hop = 0:
    predictions = observed affinities
5. For hop = 1:
    propagate affinity from neighbors:
    new_predictions[i,j] = weighted_average(neighbors_i have for j)
6. For hop = 2:
    propagate from neighbors of neighbors
    ...

7. Blend all hops with learned weights:
    final_predictions = w0 * hop0 + w1 * hop1 + w2 * hop2 + ...
```

**Hyperparameters**:
```yaml
model:
  name: mixhop_propagation
  top_k: 8                        # Keep 8 nearest neighbors
  hop_weights: [0.5, 0.3, 0.2]  # Blend 3 hops: 50%, 30%, 20%
```

**Use**: For datasets with structure (many drugs, many targets)

#### 4) InteractionCrossAttentionDTIPredictor (Advanced)

**Idea**: Use cross-attention mechanism to learn which drug/target features matter most for each interaction.

**Key Concept**: Attention weights tell the model "focus on this part of the drug SMILES when predicting for this target".

**Hyperparameters**:
```yaml
model:
  name: interaction_cross_attention
  latent_dim: 32
  epochs: 100
  lr: 0.01
  seed: 42
  attention_temperature: 1.0      # Controls softmax sharpness
  top_k: 8                        # Optional nearest-neighbor filtering
```

**Use**: When you want interpretable predictions with attention maps

### Model Selection Logic (from config)

```python
def create_predictor(model_config):
    name = model_config.get("name", "simple_baseline")
    
    if name == "simple_baseline":
        return SimpleMatrixDTIPredictor()
    elif name == "matrix_factorization":
        return MatrixFactorizationDTIPredictor(
            latent_dim=model_config.get("latent_dim", 32),
            epochs=model_config.get("epochs", 100),
            lr=model_config.get("lr", 0.01),
            seed=model_config.get("seed", 42)
        )
    elif name == "mixhop_propagation":
        return MixHopPropagationDTIPredictor(...)
    elif name == "interaction_cross_attention":
        return InteractionCrossAttentionDTIPredictor(...)
```

---

## STAGE 6: Evaluation

**File:** `src/c2dti/evaluation.py`

### What Happens?

Predictions are compared against ground truth (known affinities) using standard metrics.

### Metrics Computed:

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|-----------------|
| **MSE** | mean((y_true - y_pred)²) | [0, ∞) | Lower is better; penalizes large errors heavily |
| **RMSE** | √MSE | [0, ∞) | Same as MSE but in original units |
| **Pearson** | Correlation coefficient | [-1, 1] | +1 = perfect linear agreement; 0 = no correlation |
| **Spearman** | Rank correlation | [-1, 1] | Like Pearson but for relative ordering |
| **CI** | Concordance Index | [0, 1] | Fraction of pairs ranked correctly; primary DTI metric |

### CI (Concordance Index) - Most Important

**What**: Given two random drug-target pairs with affinities A and B (where A > B), what fraction of time does your model predict A > B?

**Formula**:
```
CI = (# of correctly ordered pairs) / (# of pairs evaluated)
```

**Range**: [0, 1]
- CI = 1.0 : Perfect ranking
- CI = 0.5 : Random guessing
- CI = 0.0 : Reverse ranking

**Why Important**: DTI literature (MINDG, DeepDTA, etc.) uses CI as the primary benchmark.

### Usage:

```python
from src.c2dti.evaluation import evaluate_predictions

metrics = evaluate_predictions(y_true, y_pred)
# Returns dict: {"mse": ..., "rmse": ..., "pearson": ..., "spearman": ..., "ci": ...}

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"CI: {metrics['ci']:.4f}")
```

### When Evaluation Happens:

```python
# If split was performed:
y_true_test = dataset.interactions[test_mask]  # Only test pairs
y_pred_test = predictions[test_mask]
metrics = evaluate_predictions(y_true_test, y_pred_test)

# Also evaluate training set to detect overfitting:
y_true_train = dataset.interactions[train_mask]
y_pred_train = predictions[train_mask]
train_metrics = evaluate_predictions(y_true_train, y_pred_train)

# If no split (backward compatible):
metrics = evaluate_predictions(dataset.interactions, predictions)  # All pairs
```

---

## Complete Data Flow Example

### Starting from YAML Config:

```yaml
# configs/davis_real_pipeline.yaml
name: C2DTI_DAVIS_REAL_PIPELINE
protocol: P1
dataset:
  name: DAVIS
  path: data/davis
  allow_placeholder: false

model:
  name: matrix_factorization
  latent_dim: 32
  epochs: 100
  lr: 0.01

split:
  strategy: "cold_drug"
  test_ratio: 0.2
  seed: 42
```

### Step-by-step execution:

```
1. LOAD CONFIG
   └─ Parse YAML, validate all fields

2. STAGE 1: LOAD DATA
   └─ Call: load_dti_dataset("DAVIS", Path("data/davis"))
   └─ DAVISLoader reads davis.csv
   └─ Returns DTIDataset with 445 drugs, 672 targets, 30k interactions

3. STAGE 2: CLEAN DATA (built-in)
   └─ Normalize column names
   └─ Coerce types
   └─ Drop NaN entries
   └─ Deduplicate sequences

4. STAGE 3: FEATURE ENGINEERING
   └─ build_string_feature_matrix(drugs)      → (445, 16)
   └─ build_string_feature_matrix(targets)    → (672, 16)

5. STAGE 4: SPLIT
   └─ Call: split_dataset(dataset, strategy="cold_drug", test_ratio=0.2)
   └─ Randomly hold out 89 drugs (20% of 445) for testing
   └─ All interactions of held-out drugs → test_mask
   └─ Remaining interactions → train_mask
   └─ Result: ~24k train pairs, ~6k test pairs

6. STAGE 5: TRAIN & PREDICT
   └─ Create MatrixFactorizationDTIPredictor(latent_dim=32, epochs=100, ...)
   └─ Call: predictor.fit_predict(dataset, train_mask=train_mask)
   
   Inside fit_predict:
     a. Initialize P (445×32) and Q (672×32) with random values
     b. For epoch 1 to 100:
          - Compute predictions = P @ Q.T    (445×672)
          - Compute error only on train_mask (never look at test!)
          - Compute gradients
          - Update P and Q
          - Log loss
     c. Apply sigmoid: predictions = 1 / (1 + exp(-predictions))
     d. Return predictions (445×672), values in [0,1]

7. STAGE 6: EVALUATE
   └─ Extract test pairs: y_true_test = interactions[test_mask]
   └─ Extract test predictions: y_pred_test = predictions[test_mask]
   └─ Call: evaluate_predictions(y_true_test, y_pred_test)
   └─ Compute MSE, RMSE, Pearson, Spearman, CI on ~6k test pairs
   └─ Example result: {"mse": 0.123, "ci": 0.78}

8. OUTPUT
   └─ Write predictions to CSV
   └─ Write metrics to JSON summary
   └─ Save model checkpoint (embeddings)
```

---

## Configuration File Structure

### Minimal Config (dry-run only):

```yaml
name: C2DTI_MINIMAL
protocol: P0
output:
  base_dir: outputs
```

### Dataset Pipeline Config (with dataset):

```yaml
name: C2DTI_DAVIS_REAL_PIPELINE
protocol: P1
output:
  base_dir: outputs

dataset:
  name: DAVIS                    # or KIBA, BindingDB
  path: data/davis               # Directory for DAVIS/KIBA; CSV file for BindingDB
  allow_placeholder: false       # Fail if data files missing

model:
  name: simple_baseline          # or matrix_factorization, mixhop_propagation, ...
  latent_dim: 32                # (optional) for matrix_factorization
  epochs: 100                   # (optional) for trainable models
  lr: 0.01                      # (optional)
  seed: 42                      # (optional)

split:                          # (optional) if omitted, train on all pairs
  strategy: random              # or cold_drug, cold_target
  test_ratio: 0.2               # 20% to test, 80% to train
  seed: 42

perturbation:                   # (optional) for causal robustness
  strength: 0.2                 # Perturbation magnitude
  seed: 42

causal:                         # (optional)
  enabled: true
  weight: 1.0
```

---

## Key Data Structures

### DTIDataset

```python
@dataclass
class DTIDataset:
    drugs: List[str]                    # e.g., ["CC(C)Cc1cc...", "CCc1cc(OC)..."]
    targets: List[str]                  # e.g., ["MQSQH...", "MSTQ..."]
    interactions: np.ndarray            # shape (n_drugs, n_targets), dtype float32
    metadata: Dict[str, Any]            # {"source": "DAVIS", "n_drugs": 445, ...}
```

### Masks (boolean arrays)

```python
train_mask: np.ndarray of shape (n_drugs, n_targets)   # True if trainable
test_mask: np.ndarray of shape (n_drugs, n_targets)    # True if test pair
known_mask: np.ndarray of shape (n_drugs, n_targets)   # True if known affinity
```

### Predictions

```python
predictions: np.ndarray of shape (n_drugs, n_targets)  # Values in [0, 1]
                                                       # dtype float32
```

---

## Common Tasks

### Load DAVIS and inspect

```python
from pathlib import Path
from src.c2dti.dataset_loader import load_dti_dataset

ds = load_dti_dataset("DAVIS", Path("data/davis"))
print(f"Drugs: {len(ds.drugs)}")
print(f"Targets: {len(ds.targets)}")
print(f"Interactions shape: {ds.interactions.shape}")
print(f"Known pairs: {(~np.isnan(ds.interactions)).sum()}")
```

### Perform random train/test split

```python
from src.c2dti.splitter import split_dataset

train_mask, test_mask = split_dataset(
    ds,
    strategy="random",
    test_ratio=0.2,
    seed=42
)

print(f"Train pairs: {train_mask.sum()}")
print(f"Test pairs: {test_mask.sum()}")
```

### Train a simple model

```python
from src.c2dti.dti_model import MatrixFactorizationDTIPredictor

model = MatrixFactorizationDTIPredictor(
    latent_dim=32,
    epochs=50,
    lr=0.01,
    seed=42
)

predictions = model.fit_predict(ds, train_mask=train_mask)
print(f"Predictions shape: {predictions.shape}")
print(f"Loss after training: {model.train_loss_history[-1]:.4f}")
```

### Evaluate predictions

```python
from src.c2dti.evaluation import evaluate_predictions

y_true = ds.interactions[test_mask]
y_pred = predictions[test_mask]

metrics = evaluate_predictions(y_true, y_pred)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"CI: {metrics['ci']:.4f}")
print(f"Pearson: {metrics['pearson']:.4f}")
```

### Run full pipeline from config

```python
python scripts/run.py --config configs/davis_real_pipeline.yaml --run-once
```

This:
1. Loads config
2. Loads dataset
3. Creates train/test split
4. Trains model
5. Evaluates
6. Saves results to `outputs/` directory

---

## Hyperparameters Summary

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `latent_dim` | int | 32 | Embedding size (matrix factorization) |
| `epochs` | int | 100 | Training iterations |
| `lr` | float | 0.01 | Learning rate (gradient descent step size) |
| `seed` | int | 42 | Random seed (reproducibility) |
| `test_ratio` | float | 0.2 | Fraction of pairs for testing |
| `top_k` | int | 8 | Nearest neighbors (MixHop, attention) |
| `attention_temperature` | float | 1.0 | Softmax sharpness (attention) |
| `hop_weights` | list | [0.5, 0.3, 0.2] | Blend weights (MixHop) |
| `threshold_pkd` | float | 7.6 | Binarization threshold (BindingDB) |

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/c2dti/dataset_loader.py` | Load DAVIS/KIBA/BindingDB → standardized DTIDataset |
| `src/c2dti/data_utils.py` | Feature engineering (build_string_feature_matrix) |
| `src/c2dti/splitter.py` | Train/test splits (random, cold_drug, cold_target) |
| `src/c2dti/dti_model.py` | Four predictor implementations |
| `src/c2dti/evaluation.py` | Metrics (MSE, RMSE, Pearson, Spearman, CI) |
| `src/c2dti/data_check.py` | Dataset validation and diagnostics |
| `src/c2dti/runner.py` | Main orchestrator (loads config, runs pipeline) |
| `src/c2dti/config_validation.py` | YAML config validation |
| `scripts/run.py` | Entry point for users |

---

## Next Steps for Causal Integration

For causal learning work (separate from this pipeline):

1. **Perturbation**: `src/c2dti/perturbation.py` adds noise to interactions
2. **Causal scoring**: `src/c2dti/causal_objective.py` computes reliability scores
3. **Full run with causal**: Set `causal.enabled: true` in config

These run *after* the core pipeline completes.
