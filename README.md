# Model Structure and Data Pipeline Overview

Run command:
```
sbatch ClimaX_baseline/run_climax_full.sbatch
```

This document summarizes the model structure, data tokenization, and DataLoader design in the current implementation.

## 1. Data Format and ClimaX Adaptation Principles

### 1.1 Why we do not force the original ClimaX input format
ClimaX originally assumes “one variable = one global image,” i.e., each variable is a full global raster field.  
If we forcibly reshape our current data into that image-like format, it would introduce several problems:
- **Structural mismatch**: time series and vertical structures (e.g., soil profiles, PFT dimensions) would be flattened or merged, causing semantic confusion.  
- **Alignment complexity**: different modalities have different temporal resolutions, missing patterns, and spatial coverage; hard alignment amplifies noise and missingness.  
- **Pipeline incompatibility**: the existing data loader and training pipeline would need a major rewrite, with high risk and cost.  
Overall, forcing the original ClimaX input format is likely to significantly degrade model performance.  
Therefore, we **keep the current dataset format and loader** while **implementing the core ClimaX ideas**.

### 1.2 Our adaptation strategy
We preserve the data format and apply ClimaX’s “variable-centric” ideas inside the model:
- **Keep multi-modal inputs**: continue using time series, static, PFT parameters, scalar, 1D PFT, and 2D soil inputs.  
- **Variable-level modeling**: each modality is encoded into a single variable-level token.  
- **Variable-level aggregation**: a learnable query aggregates variable tokens into a global representation.  
- **Downstream prediction**: the aggregated representation is fed into the rest of the network for prediction.  

### 1.3 Group-based token aggregation vs. original ClimaX idea
Original ClimaX aggregates multiple variable tokens **within the same spatial patch**, emphasizing variable-level fusion.  
Our adaptation **groups multiple variables into one token**, which is a direct transfer of the same principle:  
- Original: tokens are aggregated per patch before higher-level modeling.  
- Here: tokens are aggregated per sample before higher-level modeling.  
Both reflect the design principle of “variable-level integration first, high-level modeling later.”

## 2. Model Structure (ClimaX Full)

Core flow:
- **Multi-modal independent encoding (Variable Tokenization)**
  - Time series, static, PFT parameters, scalar, 1D PFT, and 2D soil are each encoded into a variable-level token.
- **Variable Embedding**
  - Each variable token receives a learnable identity embedding.
- **Variable Aggregation**
  - A learnable query aggregates all variable tokens via attention (the ClimaX core).
- **ViT Blocks**
  - The aggregated token goes through ViT blocks and LayerNorm.
- **Output Heads**
  - Reuse the original CNP multi-head outputs: scalar / soil_2d / pft_1d (water optional).

## 2.1 Three Core Ideas of ClimaX (Text-Only)

These are the three key ideas, rewritten as conceptual descriptions without code details.

### 1) Variable Tokenization
- **Meaning**: treat each modality (time series, static, PFT parameters, scalar, 1D PFT, 2D soil) as a separate “variable,” and encode each into one token.  
- **Purpose**: explicitly separate variable types at the start, avoiding premature mixing of different structures and scales.  
- **Effect**: each variable becomes an independent information unit for later fusion.

### 2) Variable Embedding
- **Meaning**: attach a dedicated identity embedding to each variable token.  
- **Purpose**: let the model know which variable each token represents.  
- **Effect**: improves distinguishability and stability even when numeric ranges are similar.

### 3) Variable Aggregation (Most Critical)
- **Meaning**: use a learnable global query to aggregate variable tokens through cross-attention, producing a single global representation.  
- **Purpose**: integrate information at the variable level rather than mixing all tokens in a generic self-attention block.  
- **Effect**:  
  - emphasizes variable importance through attention weights;  
  - reduces cross-variable interaction complexity;  
  - aligns with ClimaX’s variable-centric design.

## 3. Data Tokenization Design

### 3.1 Time series tokenization
- Time series are segmented by fixed windows and encoded into token sequences.  
- Temporal positions and geo-location information are included for spatiotemporal consistency.  
- The result is a time-series representation that can be combined with other modalities.

### 3.2 Static tokenization
- Static variables are grouped by semantics before tokenization.  
- This allows correlated variables to be modeled together while preserving single-variable information.

### 3.3 Variable-level tokenization (ClimaX Full)
- Each modality is treated as a variable and reduced to one token.  
- Tokens are aggregated first, then fed into higher-level Transformers.

### 3.4 List data normalization
- 1D lists are unified to a fixed length with trimming or padding.  
- 2D soil data are unified to fixed depth/layer shapes for batch consistency.

## 4. DataLoader Design

### 4.1 Data loading
- Supports multiple paths and file patterns, loaded and merged into a single dataset.  
- Allows limiting file counts for quick experiments.

### 4.2 Preprocessing
- Supports rule-based sample/variable filtering.  
- Time series are standardized to fixed length with missing values handled.  
- List-like data are reshaped to consistent formats for stable batching.  
- The dataset is shuffled before training to improve generalization.

### 4.3 Normalization strategies
- group: normalize variables by type as a group.  
- individual: normalize each variable independently to avoid scale compression.  
- hybrid: mix group and individual normalization by variable type.  

Outputs include inputs/targets and normalization metadata to keep training and inference consistent.

### 4.4 Train/validation split
- Splits data by configured ratios.  
- Returns standardized train/validation structures ready for training.


