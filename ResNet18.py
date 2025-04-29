import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18

# Load ResNet
class ResNetForClassification(nn.Module):
    def __init__(self, num_classes):
        super(ResNetForClassification, self).__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet-18
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)  # Update for num_classes

    def forward(self, x):
        return self.base_model(x)

# Data Loading
def get_dataloaders(data_dir, batch_size):
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    print("Class-to-Index Mapping:", train_dataset.class_to_idx)
    return train_loader, test_loader

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if (epoch +1)%100 ==0:
                print(f"Training - Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (epoch +1)%100 ==0:
                print(f"Batch Loss: {loss.item():.4f}")
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%, "
              f"Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {100.*test_correct/test_total:.2f}%")

        # Save Model after each epoch
        model_save_path = f"trained_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path} after epoch {epoch+1}")

# Main Script
if __name__ == "__main__":
    data_dir = "dataset"
    batch_size = 16
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(data_dir, batch_size)

    # Initialize ResNet-based model
    num_classes = 4
    model = ResNetForClassification(num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

    # Save final model
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
