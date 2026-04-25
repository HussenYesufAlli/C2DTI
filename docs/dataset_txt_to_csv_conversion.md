# Dataset TXT-to-CSV Conversion Process

## Overview
Converted DAVIS and KIBA datasets from separate text files to unified flat CSV format with 5 columns: `Drug_ID, Drug, Target_ID, Target, Y`

## Original Format (Text Files)
- **drug_smiles.txt**: One SMILES string per line (N_drugs rows)
- **target_sequences.txt**: One amino acid sequence per line (N_targets rows)
- **Y.txt**: Tab/space-separated matrix (N_drugs rows × N_targets cols)

## Target Format (CSV)
Single flat CSV with:
- `Drug_ID`: Integer positional index (0 to n_drugs-1)
- `Drug`: SMILES string
- `Target_ID`: Integer positional index (0 to n_targets-1)
- `Target`: Amino acid sequence
- `Y`: Affinity value

## Conversion Code
```python
import numpy as np
import pandas as pd
from pathlib import Path

for name in ['davis', 'kiba']:
    d = Path(f'data/{name}')
    
    # Read original text files
    drugs   = [l.strip() for l in open(d/'drug_smiles.txt') if l.strip()]
    targets = [l.strip() for l in open(d/'target_sequences.txt') if l.strip()]
    Y = np.loadtxt(d/'Y.txt', dtype=np.float32)
    
    # Melt into flat rows: iterate all drug×target pairs
    rows = []
    for i, smiles in enumerate(drugs):
        for j, seq in enumerate(targets):
            rows.append({
                'Drug_ID': i,
                'Drug': smiles,
                'Target_ID': j,
                'Target': seq,
                'Y': float(Y[i, j])
            })
    
    # Save to CSV
    df = pd.DataFrame(rows, columns=['Drug_ID','Drug','Target_ID','Target','Y'])
    out = d / f'{name}.csv'
    df.to_csv(out, index=False)
    print(f'{name}: {len(df)} rows -> {out}')
```

## Result
- **DAVIS**: 25,772 rows (68 drugs × 379 targets)
- **KIBA**: 473,572 rows (2,068 drugs × 229 targets)

## Key Points
1. **Preserves order**: `Drug_ID` = original line number in `drug_smiles.txt`, `Target_ID` = original line number in `target_sequences.txt`
2. **All pairs included**: Every drug×target combination, even if affinity was 0
3. **Matrix reconstruction**: Loader reverses process using `Drug_ID` and `Target_ID` to rebuild matrix

## Files Generated
- `data/davis/davis.csv` — 25,772 rows (full flat format)
- `data/kiba/kiba.csv` — 473,572 rows (full flat format)

## Loader Implementation
See [dataset_loader.py](../src/c2dti/dataset_loader.py):
- `_MatrixCSVLoader` — Base class that reads flat CSV and reconstructs matrix
- `DAVISLoader` — Inherits from `_MatrixCSVLoader`, loads `data/davis/davis.csv`
- `KIBALoader` — Inherits from `_MatrixCSVLoader`, loads `data/kiba/kiba.csv`
