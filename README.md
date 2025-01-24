# PyTorch to Burn Framework Migration Issue

This repository contains a minimal reproducible example demonstrating issues
when migrating a PyTorch binary classification model to the Burn framework in
Rust.

## Structure

- Python implementation: Binary classifier with 3 layers (input->64->32->1)
- Rust/Burn implementation: Attempted to mirror the PyTorch architecture
- Test suites to verify data preprocessing and loss calculations

## Commands and Outputs

### Data Processing Verification

Check if data normalization/scaling matches between Python and Rust:

```bash
# Python
python test_scaler.py

OUTPUT:
=== Training Data Check (Python) ===

Dataset sizes:
Train: 14690 Test: 3673

Feature statistics (first 3):

Feature 0 (feature1):
Mean:   0.0029
StdDev: 1.0035
Min:    -2.1987
Max:    5.7738

Feature 1 (feature2):
Mean:   -0.0012
StdDev: 0.9764
Min:    -1.3040
Max:    17.6505

Feature 2 (feature3):
Mean:   0.0044
StdDev: 1.0126
Min:    -0.7085
Max:    7.9158

Sample values (first 3 samples, first 3 features):
Sample 0: [-1.00928936 -0.86094706 -0.40131254]
Sample 1: [-0.49493203  0.25712502  2.84591398]
Sample 2: [ 0.05157263  0.06726372 -0.22105967]

# Rust
cargo run -- test-training-data

OUTPUT:
=== Training Data Check (Rust) ===

Dataset sizes:
Train: 14690 Test: 3673

Feature statistics (first 3):

Feature 0:
Mean:   -0.0034
StdDev: 1.0025
Min:    -2.1987
Max:    5.7738

Feature 1:
Mean:   0.0006
StdDev: 0.9915
Min:    -1.3040
Max:    21.7539

Feature 2:
Mean:   0.0012
StdDev: 1.0048
Min:    -0.7085
Max:    7.9158

Sample values (first 3 samples, first 3 features):
Sample 0: [0.2766045, 0.37315604, -0.57968134]
Sample 1: [-0.6556703, -0.36520272, -0.28014097]
Sample 2: [1.0481423, 0.78452736, -0.44031748]
```

### BCE Loss Function Test

Verify BCE loss calculation between frameworks:

```bash
# Python
python test_bce_loss.py

OUTPUT:
Test Case 1 - Perfect predictions:
Predictions: tensor([1., 0., 1., 0.])
Targets:     tensor([1., 0., 1., 0.])
Loss:        0.00000000

Test Case 2 - Wrong predictions:
Predictions: tensor([0., 1., 0., 1.])
Targets:     tensor([1., 0., 1., 0.])
Loss:        100.00000000

Test Case 3 - Uncertain predictions:
Predictions: tensor([0.5000, 0.5000, 0.5000, 0.5000])
Targets:     tensor([1., 0., 1., 0.])
Loss:        0.69314718

# Rust 
cargo run -- test-bce-loss

OUTPUT:
Test Case 1 - Perfect predictions:
Predictions: tensor([1.0000, 0.0000, 1.0000, 0.0000])
Targets:     tensor([1, 0, 1, 0])
Loss:        NaN

Test Case 2 - Wrong predictions:
Predictions: tensor([0.0000, 1.0000, 0.0000, 1.0000])
Targets:     tensor([1, 0, 1, 0])
Loss:        inf

Test Case 3 - Uncertain predictions:
Predictions: tensor([0.5000, 0.5000, 0.5000, 0.5000])
Targets:     tensor([1, 0, 1, 0])
Loss:        0.69314718
```

### Label Distribution Check

```bash
cargo run -- test-label-percentage

OUTPUT:
Train distribution: total=14690, zeroes=6742 (45.9%), ones=7948 (54.1%)
Test distribution: total=3673, zeroes=1713 (46.6%), ones=1960 (53.4%)
```

### Training

```bash
python pytorchml.py

OUTPUT:
Epoch 1/20, Loss: 0.41438834268761715
Epoch 2/20, Loss: 0.2651732108231796
Epoch 3/20, Loss: 0.22890679868667022
Epoch 4/20, Loss: 0.20923193655503186
Epoch 5/20, Loss: 0.19471799664037384
Epoch 6/20, Loss: 0.18624816523136004
Epoch 7/20, Loss: 0.17845469967907537
Epoch 8/20, Loss: 0.17498739161629878
Epoch 9/20, Loss: 0.16884841593488564
Epoch 10/20, Loss: 0.1632193924251782
Epoch 11/20, Loss: 0.16073451330480368
Epoch 12/20, Loss: 0.15668279322714584
Epoch 13/20, Loss: 0.157244895003817
Epoch 14/20, Loss: 0.1534707573444947
Epoch 15/20, Loss: 0.15048636633204612
Epoch 16/20, Loss: 0.147686033612927
Epoch 17/20, Loss: 0.1458114351996261
Epoch 18/20, Loss: 0.14404122283803705
Epoch 19/20, Loss: 0.14307642775263799
Epoch 20/20, Loss: 0.14222166280998624
Model saved to data/pytorch/demo_classifier.pth
Accuracy: 93.36%

cargo run -- train

OUTPUT:
Model:
DemoClassifierModel {
  input_layer: Linear {d_input: 20, d_output: 64, bias: true, params: 1344}
  hidden_layer1: Linear {d_input: 64, d_output: 32, bias: true, params: 2080}
  output_layer: Linear {d_input: 32, d_output: 1, bias: true, params: 33}
  activation: Relu
  sigmoid: Sigmoid
  params: 3457
}
Total Epochs: 20


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 46.113   | 1        | 46.113   | 20       |
| Train | Loss     | 0.279    | 10       | NaN      | 20       |
| Valid | Accuracy | 45.766   | 1        | 45.766   | 20       |
| Valid | Loss     | 0.282    | 10       | NaN      | 20       |
```

### Critical Code Section

```rust
// In src/training.rs
// Original implementation
let loss = BinaryCrossEntropyLossConfig::new()
    .init(&output.device())
    .forward(output.clone().squeeze(1), targets.clone());

// Alternative attempt
// let loss = BinaryCrossEntropyLossConfig::new()
//     .init(&output.device())
//     .forward(output.clone(), targets.clone().reshape([batch_size, 1]));
```

## Environment

- Python 3.8+
- PyTorch 2.0+
- Rust 1.75+
- Burn 0.16.0
