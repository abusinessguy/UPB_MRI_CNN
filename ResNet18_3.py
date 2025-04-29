import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import csv
import numpy as np

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
    train_transform = Compose([
        Resize((256, 256)),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    test_dataset = ImageFolder(root=f"{data_dir}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    print("Class-to-Index Mapping:", train_dataset.class_to_idx)
    return train_loader, test_loader, train_dataset

# Freeze Layers Function
def freeze_layers(model, num_layers_to_freeze):
    child_counter = 0
    for child in model.base_model.children():
        if child_counter < num_layers_to_freeze:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
        child_counter += 1

# Training Function with tqdm and CSV Logging
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler):
    model.to(device)
    metrics = []  # To store metrics for CSV output
    
    for epoch in range(num_epochs):
        # Freeze the first two layers for all but the last 5 epochs
        if epoch < num_epochs - 5:
            freeze_layers(model, 2)
        else:
            freeze_layers(model, 0)  # Unfreeze all layers
        
        model.train()  # Ensure BatchNorm and Dropout layers behave correctly
        train_loss, train_correct, train_total = 0, 0, 0

        # Training loop with tqdm progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_bar.set_postfix(loss=train_loss / (train_total or 1))

        train_acc = 100. * train_correct / train_total

        # Validation loop
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing")
            for images, labels in test_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        # Adjust learning rate based on test loss
        scheduler.step(test_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_acc:.2f}%")

        # Save metrics for CSV
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'test_loss': test_loss / len(test_loader),
            'test_acc': test_acc
        })

        # Save Model after each epoch
        # model_save_path = f"trained_model_epoch_{epoch+1}.pth"
        # torch.save(model.state_dict(), model_save_path)
        # print(f"Model saved to {model_save_path} after epoch {epoch+1}")

    # Write metrics to CSV
    csv_file = "training_metrics.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Training metrics saved to {csv_file}")

# Main Script
if __name__ == "__main__":
    data_dir = "dataset"
    batch_size = 32
    num_epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, test_loader, train_dataset = get_dataloaders(data_dir, batch_size)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.arange(len(train_dataset.classes)), y=train_dataset.targets)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize ResNet-based model
    num_classes = len(train_dataset.classes)
    model = ResNetForClassification(num_classes=num_classes)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, scheduler)

    # Save final model
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
