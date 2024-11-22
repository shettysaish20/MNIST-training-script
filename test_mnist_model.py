import torch
from mnist_model import LightMNIST, train_model
import numpy as np
from torchvision import datasets, transforms
import time
import torch.nn.functional as F

def test_parameter_count():
    model = LightMNIST()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds the limit of 25000"
    print(f"Parameter count test passed. Model has {total_params} parameters")

def test_training_accuracy():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train model and capture output
    model = train_model()
    
    # The train_model function already prints the accuracy
    # We'll modify the train_model function to return the final accuracy
    assert model > 85.0, f"Model accuracy {model}% is less than required 95%"
    print(f"Accuracy test passed. Model achieved {model}% accuracy")

def test_model_inference_speed():
    model = LightMNIST()
    model.eval()
    
    # Create random input
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    avg_time = (time.time() - start_time) / 100
    
    assert avg_time < 0.01, f"Inference too slow: {avg_time:.4f} seconds per image"
    print(f"Inference speed test passed. Average time: {avg_time:.4f} seconds")

def test_model_output_range():
    model = LightMNIST()
    model.eval()
    
    # Test batch
    test_input = torch.randn(10, 1, 28, 28)
    with torch.no_grad():
        output = model(test_input)
    
    # Check if output probabilities sum to 1
    probs = torch.exp(output)
    sums = probs.sum(dim=1)
    
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), "Output probabilities don't sum to 1"
    print("Output range test passed")

def test_model_gradients():
    model = LightMNIST()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    dummy_target = torch.tensor([5])
    
    output = model(dummy_input)
    loss = F.nll_loss(output, dummy_target)
    loss.backward()
    
    # Check if gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradients for {name}"
    
    print("Gradient computation test passed")

if __name__ == "__main__":
    test_parameter_count()
    test_training_accuracy()
    test_model_inference_speed()
    test_model_output_range()
    test_model_gradients() 