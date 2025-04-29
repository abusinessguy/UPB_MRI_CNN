import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from ResNet18 import ResNetForClassification  # Your custom model import

def generate_confusion_matrix(model, dataloader, device, class_names):
    """
    Generate and display a confusion matrix for a trained model on a test dataset.

    :param model: Trained PyTorch model.
    :param dataloader: DataLoader for the test dataset.
    :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :param class_names: List of class names.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'.")

def generate_binary_confusion_matrix(model, dataloader, device, class_names, tumor_classes):
    """
    Generate and display a binary confusion matrix for tumor and no tumor categories.

    :param model: Trained PyTorch model.
    :param dataloader: DataLoader for the test dataset.
    :param device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    :param class_names: List of class names.
    :param tumor_classes: List of class indices corresponding to tumor classes.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Map class indices to binary categories
    tumor_class_set = set(tumor_classes)
    binary_preds = [1 if pred in tumor_class_set else 0 for pred in all_preds]
    binary_labels = [1 if label in tumor_class_set else 0 for label in all_labels]

    # Generate confusion matrix
    cm = confusion_matrix(binary_labels, binary_preds, labels=[0, 1])

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Tumor", "Tumor"])
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Binary Confusion Matrix (Tumor vs No Tumor)")
    plt.tight_layout()
    plt.savefig("binary_confusion_matrix.png")
    print("Binary confusion matrix saved as 'binary_confusion_matrix.png'.")


if __name__ == "__main__":
    # Load your trained model
    model = ResNetForClassification(num_classes=4)  # Adjust num_classes as needed
    model.load_state_dict(torch.load("C:/Projects/MRI_CNN/trained_model.pth"))
    model.eval()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset
    test_dir = "C:/Projects/MRI_CNN/dataset/test"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjusted to match training normalization
    ])
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Class names
    class_names = test_dataset.classes
    print(class_names)

    # Generate the confusion matrix
    generate_confusion_matrix(model, test_loader, device, class_names)

    # Define tumor classes (indices in class_names list corresponding to tumor classes)
    tumor_classes = [0, 1, 3]  # Replace with actual indices for tumor classes

    # Generate the binary confusion matrix
    generate_binary_confusion_matrix(model, test_loader, device, class_names, tumor_classes)