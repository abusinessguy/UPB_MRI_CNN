import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

# Custom Model for Classification
class UNetForClassification(nn.Module):
    def __init__(self, base_model, num_classes):
        super(UNetForClassification, self).__init__()
        # Use pre-trained encoder layers
        self.encoder1 = base_model.encoder1
        self.encoder2 = base_model.encoder2
        self.encoder3 = base_model.encoder3
        self.encoder4 = base_model.encoder4
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global pooling
        self.fc = nn.Linear(256, num_classes)  # Update input size to 256

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        # print(f"After encoder4: {x.shape}")  # Debug: Expected (batch_size, 256, height, width)
        x = self.global_pool(x).view(x.size(0), -1)  # Pool and flatten
        # print(f"After global pool: {x.shape}")  # Debug: Expected (batch_size, 256)
        x = self.fc(x)  # Fully connected classification
        # print(f"After fully connected: {x.shape}")  # Debug: Expected (batch_size, num_classes)
        return x

# Data Loading
def get_dataloaders(data_dir, batch_size):
    # Define transformations
    transform = Compose([
        Resize((256, 256)),       # Resize to match model input
        ToTensor(),               # Convert to tensor
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB images
    ])

    # Load datasets
    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=transform)
    test_dataset = ImageFolder(root=f"{data_dir}/test", transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print("Class-to-Index Mapping:", train_dataset.class_to_idx)
    return train_loader, test_loader

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Training - Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}]")
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Shape: (batch_size, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(f"Batch Loss: {loss.item():.4f}")
            _, predicted = outputs.max(1)  # Predicted class
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation phase
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

        # Save Model
        model_save_path = f"trained_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path} after epoch {epoch+1}")

# Main Script
if __name__ == "__main__":
    # Path to your dataset
    data_dir = "dataset"  # Ensure dataset/train and dataset/test exist
    batch_size = 16
    num_epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    train_loader, test_loader = get_dataloaders(data_dir, batch_size)

    # Load pre-trained U-Net
    pretrained_unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                     in_channels=3, out_channels=1, init_features=32, pretrained=True)

    # Initialize classification model
    model = UNetForClassification(pretrained_unet, num_classes=4)

    # Freeze encoder layers
    for param in model.encoder1.parameters():
        param.requires_grad = False
    for param in model.encoder2.parameters():
        param.requires_grad = False
    for param in model.encoder3.parameters():
        param.requires_grad = False
    for param in model.encoder4.parameters():
        param.requires_grad = False

    print("Froze encoder layers.")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

    # Save the trained model
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # # Unfreeze encoder layers for fine-tuning (optional, after initial training)
    # for param in model.encoder1.parameters():
    #     param.requires_grad = True
    # for param in model.encoder2.parameters():
    #     param.requires_grad = True
    # for param in model.encoder3.parameters():
    #     param.requires_grad = True
    # for param in model.encoder4.parameters():
    #     param.requires_grad = True

    # print("Unfroze encoder layers for fine-tuning.")

    # # Fine-tune the model
    # optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Reduce learning rate for fine-tuning
    # train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=5, device=device)
