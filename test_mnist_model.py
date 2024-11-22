import torch
from mnist_model import LightMNIST, train_model

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
    assert model > 95.0, f"Model accuracy {model}% is less than required 95%"
    print(f"Accuracy test passed. Model achieved {model}% accuracy")

if __name__ == "__main__":
    test_parameter_count()
    test_training_accuracy() 