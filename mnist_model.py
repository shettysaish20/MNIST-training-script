import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 8x28x28
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # 16x14x14
        self.fc = nn.Linear(16 * 7 * 7, 10)
        
    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 8x14x14
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 16x7x7
        
        # Flatten and fully connected
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                 transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    
    # Model, loss and optimizer
    model = LightMNIST().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006, betas=(0.9, 0.999))
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    
    # Training loop
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    final_accuracy = 100. * correct / total
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    return final_accuracy

if __name__ == '__main__':
    train_model() 