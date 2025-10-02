# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Overview

This repository demonstrates a minimal reproducible example of PyTorch to Burn
framework migration issues. It contains parallel implementations of a binary
classification model in both Python (PyTorch) and Rust (Burn), with the Rust
implementation exhibiting training failures (NaN/inf losses) despite matching
architecture.

## Key Commands

### How to run python scripts

You'll need to activate virtual env first:

```
source .venv/bin/activate.fish 
```

### Testing
```bash
# Rust data preprocessing verification
cargo run -- test-training-data

# Rust BCE loss function test
cargo run -- test-bce-loss

# Rust label distribution check
cargo run -- test-label-percentage

# Rust unit tests
cargo test

# Python data preprocessing verification
python test_scaler.py

# Python BCE loss function test
python test_bce_loss.py
```

### Training
```bash
# Train Rust model (exhibits NaN/inf loss issues)
cargo run -- train

# Train Python model (works correctly)
python pytorchml.py
```

## Critical Issue

The Rust implementation produces NaN/inf losses during training despite:
- Matching model architecture (input->64->32->1)
- Identical layer configurations
- Same activation functions (ReLU, Sigmoid)
- Similar data preprocessing

**Problem location:** src/training.rs:50-52

The BCE loss calculation causes NaN/inf values when predictions are exactly 0.0 or 1.0:
```rust
let loss = BinaryCrossEntropyLossConfig::new()
    .init(&output.device())
    .forward(output.clone().squeeze(1), targets.clone());
```

Alternative approach (commented out in training.rs:54-58) also fails.

## Architecture

### Data Flow
1. **Raw data** (CSV) → `RawFeatures` (src/data.rs)
2. **Feature extraction** → `as_vec()` converts 20 features to Vec<f32>
3. **Normalization** → StandardScaler applied (src/scaler.rs)
4. **Prepared data** → `PreparedFeatures` (src/prepared.rs)
5. **Batching** → `DemoBatcher` creates tensors (src/dataset.rs)
6. **Model** → `DemoClassifierModel` (src/model.rs)

### Model Architecture
- **Input layer:** 20 features → 64 (ReLU)
- **Hidden layer:** 64 → 32 (ReLU)
- **Output layer:** 32 → 1 (Sigmoid)

Mirrors PyTorch implementation in pytorchml.py with identical layer sizes and activations.

### Training Configuration (src/training.rs)
- Optimizer: Adam with gradient clipping (norm=1.0)
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 20
- Loss: Binary Cross Entropy

### Data Characteristics
- **Train set:** 14,690 samples (54.1% positive, 45.9% negative)
- **Test set:** 3,673 samples (53.4% positive, 46.6% negative)
- **Features:** 20 normalized features (mean≈0, std≈1)

## Dependencies

### Rust
- Burn: Git version from main branch (WGPU backend disabled, using ndarray)
- Backend: NdArray (WGPU compilation issues noted in commit history)

### Python
- PyTorch 2.0+
- See pyreq.txt for requirements

## Known Issues

1. **BCE Loss with extreme predictions:** Test case shows loss=NaN for perfect predictions [1.0, 0.0, 1.0, 0.0], while PyTorch handles this gracefully (loss=0.00000000)

2. **Training divergence:** Rust model accuracy stuck at ~46% (baseline), loss becomes NaN by epoch 20. Python model reaches 93.36% accuracy with stable loss convergence.

3. **Backend compatibility:** WGPU backend unable to compile (see commit c36f56a), currently using ndarray backend as workaround.

## TODO: Binary Classification Adaptation to Burn Framework

### Current Problem

Burn's `AccuracyMetric` is designed for multi-class classification and expects
output shape `[batch_size, num_classes]`. It uses `argmax(1)` to find the
predicted class. For binary classification with output shape `[batch_size, 1]`,
this doesn't work correctly.

### PyTorch Implementation Analysis
- **Model output:** `[batch_size, 1]` → squeezed to `[batch_size]` probabilities [0, 1]
- **Labels:** `[batch_size]` float32 values {0.0, 1.0}
- **BCE Loss:** Takes probabilities (post-sigmoid) directly
- **Accuracy:** Simple threshold at 0.5: `(outputs > 0.5).float() == labels`

### Burn Adaptation Strategy (Option C: Adapt to Multi-Class)
To properly use Burn's training framework with `AccuracyMetric` and `ClassificationOutput`:

1. **Change model output to 2-class format:**
   - Output layer: 32 → 2 (instead of 32 → 1)
   - Remove sigmoid activation from model
   - Return logits for both classes: `[batch_size, 2]`

2. **Use CrossEntropyLoss instead of BCE:**
   - CrossEntropy expects logits `[batch_size, num_classes]`
   - Targets remain `[batch_size]` as integer class indices {0, 1}
   - Handles softmax internally for numerical stability

3. **Labels format:**
   - Keep as `Tensor<B, 1, Int>` with values {0, 1}
   - No changes needed in dataset.rs

4. **Accuracy metric:**
   - Use Burn's `AccuracyMetric` unchanged
   - It will apply `argmax(1)` on `[batch_size, 2]` → get predicted class
   - Compare with integer targets

5. **Benefits:**
   - Works with Burn's `Learner`, `ClassificationOutput`, `AccuracyMetric`
   - Numerically stable (logits + CrossEntropy)
   - Standard multi-class pattern

6. **Files to modify:**
   - `src/model.rs`: Change output layer from 1 → 2 neurons, remove sigmoid
   - `src/training.rs`: Replace `BinaryCrossEntropyLoss` with `CrossEntropyLoss`
   - Keep everything else unchanged
