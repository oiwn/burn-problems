import torch
import torch.nn as nn

def test_bce_loss():
    # Initialize loss function
    criterion = nn.BCELoss()
    
    # Test Case 1: Perfect predictions
    predictions = torch.tensor([1.0, 0.0, 1.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = criterion(predictions, targets)
    print("\nTest Case 1 - Perfect predictions:")
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"Loss:        {loss.item():.8f}")
    
    # Test Case 2: Completely wrong predictions
    predictions = torch.tensor([0.0, 1.0, 0.0, 1.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = criterion(predictions, targets)
    print("\nTest Case 2 - Wrong predictions:")
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"Loss:        {loss.item():.8f}")
    
    # Test Case 3: Uncertain predictions (0.5)
    predictions = torch.tensor([0.5, 0.5, 0.5, 0.5])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = criterion(predictions, targets)
    print("\nTest Case 3 - Uncertain predictions:")
    print(f"Predictions: {predictions}")
    print(f"Targets:     {targets}")
    print(f"Loss:        {loss.item():.8f}")

if __name__ == "__main__":
    test_bce_loss()
